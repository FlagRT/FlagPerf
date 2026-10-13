// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Independent Toolkit microbenchmarks using the bundled XRE and XBLAS APIs.
#include <xpu/runtime.h>
#include <cublas_v2.h>
#include <newcontext.h>
#include <xblas_legacy_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <time.h>

static void check(int rc, const char* call) {
    if (rc) throw std::runtime_error(std::string(call) + " returned " + std::to_string(rc));
}
#define X(call) check((call), #call)
static uint64_t tick() {
    timespec t{};
    if (clock_gettime(CLOCK_MONOTONIC, &t)) throw std::runtime_error("clock_gettime failed");
    return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}
static std::string pci(int device) {
    // The bundled compatibility XRE reads this attribute from the current
    // context, even when devid is supplied. Select before querying identity.
    X(xpu_set_device(device));
    uint64_t address = 0;
    X(xpu_device_get_attr(&address, XPUATTR_PCI_ADDRESS, device));
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%08x:%02x:%02x.%x", unsigned(address >> 16),
             unsigned((address >> 8) & 255), unsigned((address >> 3) & 31), unsigned(address & 7));
    return buffer;
}
static void bind(int device, const std::string& expected) {
    if (pci(device) != expected) throw std::runtime_error("XRE device PCI identity mismatch before allocation");
    X(xpu_set_device(device));
}
struct Buffer {
    void* p = nullptr;
    int device;
    Buffer(int d, size_t size):device(d) { X(xpu_set_device(d)); X(xpu_malloc(&p, size)); }
    ~Buffer() { if (p) { xpu_set_device(device); xpu_free(p); } }
};
struct HostBuffer {
    void* p=nullptr; bool pinned,registered;
    HostBuffer(size_t bytes, bool pin, bool use_registration):pinned(pin),registered(pin && use_registration) {
        if (registered) {
            if(posix_memalign(&p,4096,bytes))throw std::bad_alloc();
            int rc=xpu_host_register(p,bytes,0);
            if(rc){free(p);p=nullptr;check(rc,"xpu_host_register");}
        }
        else if (pin) { X(xpu_host_alloc(&p, bytes, 0)); }
        else { p=malloc(bytes); if (!p) throw std::bad_alloc(); }
    }
    ~HostBuffer() { if (registered) {xpu_host_unregister(p);free(p);} else if (pinned) xpu_host_free(p); else free(p); }
};
struct Stream {
    XPUStream value=nullptr;
    Stream() { X(xpu_stream_create(&value)); }
    ~Stream() { if (value) xpu_stream_destroy(value); }
};
static void library(const void* symbol) {
    Dl_info info{};
    if (!dladdr(symbol, &info) || !info.dli_fname) throw std::runtime_error("cannot identify native library");
    std::cerr << "native_library=" << info.dli_fname << '\n';
}
struct Samples {
    std::vector<double> ns;
    uint64_t start=0, end=0;
    size_t api_calls_per_sample=1;
    size_t async_chunk_bytes=0;
    template<class F> void measure(F work, int warmup, int count, double minimum_seconds) {
        for (int i=0;i<warmup;i++) work();
        start=tick();
        do {
            uint64_t a=tick(); work(); uint64_t b=tick();
            if (b<=a) throw std::runtime_error("nonpositive completed-operation time");
            ns.push_back(double(b-a));
            // Bound output size as well as duration. Failure is explicit, not truncation.
            if (ns.size()>=2000000) throw std::runtime_error("sample limit reached");
        } while (ns.size()<size_t(count) || double(tick()-start)<minimum_seconds*1e9);
        end=tick();
    }
    void print(const std::string& mode, size_t bytes, int n, int warmup, size_t checked, const std::string& dtype) const {
        std::cout << std::setprecision(17) << "P800_METRIC {\"schema_version\":1,\"mode\":\"" << mode
          << "\",\"payload_bytes\":" << bytes << ",\"dimension\":" << n
          << ",\"warmup\":" << warmup << ",\"correctness\":true,\"checked_values\":" << checked
          << ",\"dtype\":\"" << dtype << "\",\"output_dtype\":\"" << (mode=="gemm" ? "FP32" : "bytes") << "\""
          << ",\"api_calls_per_sample\":" << api_calls_per_sample << ",\"async_chunk_bytes\":" << async_chunk_bytes
          << ",\"timer\":\"CLOCK_MONOTONIC completed API operation including synchronization\""
          << ",\"started_monotonic_ns\":" << start << ",\"finished_monotonic_ns\":" << end
          << ",\"samples_ns\":[";
        for (size_t i=0;i<ns.size();i++) std::cout << (i ? "," : "") << ns[i];
        std::cout << "]}" << std::endl;
    }
};
static unsigned char pattern(size_t i, unsigned seed=0) { return (i*131 + (i>>8)*17 + seed+19) & 255; }
// Two persistent workers execute both directions independently. The pair timer
// includes host rendezvous and waits until both directions actually complete.
class PairWorkers {
    std::mutex mutex;
    std::condition_variable cv;
    unsigned generation=0, finished=0;
    bool stop=false;
    std::exception_ptr error;
    std::thread first,second;
    void loop(std::function<void()> work) {
        unsigned seen=0;
        std::unique_lock<std::mutex> lock(mutex);
        while(true) {
            cv.wait(lock,[&]{return stop || generation!=seen;});
            if(stop) return;
            seen=generation; lock.unlock();
            std::exception_ptr failure;
            try {work();} catch(...) {failure=std::current_exception();}
            lock.lock(); if(failure)error=failure; ++finished; cv.notify_all();
        }
    }
public:
    PairWorkers(std::function<void()> a,std::function<void()> b):first([this,a]{loop(a);}),second([this,b]{loop(b);}) {}
    ~PairWorkers(){ {std::lock_guard<std::mutex> lock(mutex);stop=true;cv.notify_all();} first.join();second.join(); }
    void run(){std::unique_lock<std::mutex> lock(mutex);finished=0;++generation;cv.notify_all();cv.wait(lock,[&]{return finished==2;});if(error)std::rethrow_exception(error);}
};
static size_t bidirectional(int src,int dst,size_t bytes,Samples& samples,int warmup,int count,double seconds) {
    samples.api_calls_per_sample=2;
    std::vector<unsigned char> ha(bytes), hb(bytes), check_a(bytes),check_b(bytes);
    for(size_t i=0;i<bytes;i++){ha[i]=pattern(i);hb[i]=pattern(i,73);}
    Buffer a(src,bytes), b(dst,bytes), c(dst,bytes), d(src,bytes);
    X(xpu_set_device(src));X(xpu_memcpy(a.p,ha.data(),bytes,XPU_HOST_TO_DEVICE));X(xpu_wait());
    X(xpu_set_device(dst));X(xpu_memcpy(c.p,hb.data(),bytes,XPU_HOST_TO_DEVICE));X(xpu_wait());
    auto copy=[&](int from,int to,void* source,void* dest){
        X(xpu_set_device(from));X(xpu_memcpy_peer(to,dest,from,source,bytes));
        X(xpu_set_device(to));X(xpu_wait());X(xpu_set_device(from));X(xpu_wait());
    };
    {
        PairWorkers workers([&]{copy(src,dst,a.p,b.p);},[&]{copy(dst,src,c.p,d.p);});
        samples.measure([&]{workers.run();},warmup,count,seconds);
    }
    X(xpu_set_device(dst));X(xpu_memcpy(check_a.data(),b.p,bytes,XPU_DEVICE_TO_HOST));X(xpu_wait());
    X(xpu_set_device(src));X(xpu_memcpy(check_b.data(),d.p,bytes,XPU_DEVICE_TO_HOST));X(xpu_wait());
    if(check_a!=ha || check_b!=hb)throw std::runtime_error("bidirectional full-payload correctness failure");
    return 2*bytes;
}
static size_t copy_bench(const std::string& mode, int src, int dst, size_t bytes,
                       bool pinned, bool async, bool registration, size_t chunk, Samples& samples, int warmup, int count, double seconds) {
    HostBuffer host(bytes, pinned, registration);
    std::cerr << "host_memory_api=" << (pinned ? (registration ? "posix_memalign+xpu_host_register" : "xpu_host_alloc") : "malloc") << "\n";
    auto* h=static_cast<unsigned char*>(host.p);
    for (size_t i=0;i<bytes;i++) h[i]=pattern(i);
    Buffer a(src, bytes), b(mode=="p2p" ? dst : src, bytes);
    X(xpu_set_device(src));
    X(xpu_memcpy(a.p, host.p, bytes, XPU_HOST_TO_DEVICE)); X(xpu_wait());
    if (mode=="d2h") memset(host.p, 0xBE, bytes);
    Stream stream;
    if(pinned && async && chunk && bytes>chunk) {
        samples.api_calls_per_sample=(bytes+chunk-1)/chunk;
        samples.async_chunk_bytes=chunk;
    }
    auto work=[&]() {
        if (mode=="p2p") {
            X(xpu_memcpy_peer(dst,b.p,src,a.p,bytes));
            X(xpu_set_device(dst)); X(xpu_wait()); X(xpu_set_device(src)); X(xpu_wait());
        } else {
            void* to=mode=="d2h" ? host.p : b.p;
            const void* from=mode=="h2d" ? host.p : a.p;
            auto kind=mode=="h2d" ? XPU_HOST_TO_DEVICE : (mode=="d2h" ? XPU_DEVICE_TO_HOST : XPU_DEVICE_TO_DEVICE);
            if (async) {
                const size_t step=(pinned && chunk) ? std::min(chunk,bytes) : bytes;
                for(size_t offset=0;offset<bytes;offset+=step) {
                    X(xpu_memcpy_async(static_cast<unsigned char*>(to)+offset,static_cast<const unsigned char*>(from)+offset,std::min(step,bytes-offset),kind,stream.value));
                }
                X(xpu_wait(stream.value));
            }
            else { X(xpu_memcpy(to,from,bytes,kind)); X(xpu_wait()); }
        }
    };
    samples.measure(work,warmup,count,seconds);
    if (mode!="d2h") {
        X(xpu_set_device(mode=="p2p" ? dst : src));
        X(xpu_memcpy(host.p,b.p,bytes,XPU_DEVICE_TO_HOST)); X(xpu_wait());
    }
    for(size_t i=0;i<bytes;i++) if(h[i]!=pattern(i)) throw std::runtime_error("copy full-payload correctness failure at " + std::to_string(i));
    X(xpu_set_device(src));
    return bytes;
}
// Independent device-side copy. Keep the runtime memcpy comparator separately:
// it is not a valid proxy for HBM kernel read+write bandwidth on this stack.
static size_t kernel_copy(int device, size_t bytes, Samples& samples, int warmup, int count, double seconds) {
    if (bytes % sizeof(float)) throw std::runtime_error("kernel copy requires a multiple of four bytes");
    library(reinterpret_cast<const void*>(&cublasScopy));
    std::vector<unsigned char> host(bytes), readback(bytes, 0);
    for(size_t i=0;i<bytes;i++) host[i]=pattern(i);
    Buffer a(device,bytes), b(device,bytes);
    X(xpu_memcpy(a.p,host.data(),bytes,XPU_HOST_TO_DEVICE));
    X(xpu_memcpy(b.p,readback.data(),bytes,XPU_HOST_TO_DEVICE)); X(xpu_wait());
    cublasHandle_t handle=nullptr; X(cublasCreate(&handle));
    struct Guard {cublasHandle_t h;~Guard(){cublasDestroy(h);}} guard{handle};
    std::cerr << "copy_api=XBLAS cublasScopy; independent device kernel; read+write bytes=2*payload\n";
    auto work=[&]{X(cublasScopy(handle,bytes/sizeof(float),static_cast<const float*>(a.p),1,static_cast<float*>(b.p),1));X(xpu_wait());};
    samples.measure(work,warmup,count,seconds);
    X(xpu_memcpy(readback.data(),b.p,bytes,XPU_DEVICE_TO_HOST));X(xpu_wait());
    if(readback!=host)throw std::runtime_error("kernel copy full-payload correctness failure");
    return bytes;
}
static int aval(int r,int k) { return ((r*3+k*5)%7)-3; }
static int bval(int k,int c) { return ((k*2+c*3)%5)-2; }
static float afloat(int r,int k) {return float(aval(r,k))*.1234567f + float((r*11+k*7)%13)*.0000137f;}
static float bfloat(int k,int c) {return float(bval(k,c))*.2345678f + float((k*3+c*11)%17)*.0000173f;}
static uint16_t half_integer(int x) {
    // Test inputs are integers in [-3, 3], all exactly representable in FP16.
    static const uint16_t pos[]={0,0x3c00,0x4000,0x4200};
    return pos[std::abs(x)] | (x<0 ? 0x8000 : 0);
}
static void encode(void* p,size_t i,int x,const std::string& dtype) {
    if(dtype=="FP32") static_cast<float*>(p)[i]=float(x);
    else if(dtype=="INT8") static_cast<int8_t*>(p)[i]=int8_t(x);
    else if(dtype=="FP16") static_cast<uint16_t*>(p)[i]=half_integer(x);
    else { float f=float(x); uint32_t v;memcpy(&v,&f,4);static_cast<uint16_t*>(p)[i]=uint16_t(v>>16); }
}
static size_t gemm(const std::string& dtype,int device,int n,Samples& samples,int warmup,int count,double seconds) {
    library(reinterpret_cast<const void*>(&cublasCreate));
    size_t width=dtype=="FP32" ? 4 : (dtype=="INT8" ? 1 : 2), elements=size_t(n)*n;
    std::vector<unsigned char> ha(elements*width), hb(elements*width);
    for(int c=0;c<n;c++) for(int r=0;r<n;r++) {
        encode(ha.data(),size_t(c)*n+r,dtype=="INT8" ? aval(c,r) : aval(r,c),dtype);
        encode(hb.data(),size_t(c)*n+r,bval(r,c),dtype);
        if(dtype=="FP32") {
            reinterpret_cast<float*>(ha.data())[size_t(c)*n+r]=afloat(r,c);
            reinterpret_cast<float*>(hb.data())[size_t(c)*n+r]=bfloat(r,c);
        }
    }
    Buffer a(device,ha.size()), b(device,hb.size()), c(device,elements*4);
    X(xpu_memcpy(a.p,ha.data(),ha.size(),XPU_HOST_TO_DEVICE));
    X(xpu_memcpy(b.p,hb.data(),hb.size(),XPU_HOST_TO_DEVICE));
    X(xpu_wait());
    cublasHandle_t handle=nullptr;
    X(cublasCreate(&handle));
    struct Guard { cublasHandle_t h; ~Guard(){cublasDestroy(h);} } guard{handle};
    X(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
    cublasMath_t math_mode=static_cast<cublasMath_t>(-1);
    X(cublasGetMathMode(handle,&math_mode));
    std::cerr << "cublas_math_mode=" << int(math_mode) << "\n";
    if(math_mode!=CUBLAS_DEFAULT_MATH)throw std::runtime_error("unexpected XBLAS math mode");
    float alpha=1.0f,beta=0.0f;
    cudaDataType_t type=dtype=="FP32" ? CUDA_R_32F : dtype=="FP16" ? CUDA_R_16F : dtype=="BF16" ? CUDA_R_16BF : CUDA_R_8I;
    auto* ctx = dtype=="INT8" ? baidu::xpu::api::create_context() : nullptr;
    struct ContextGuard { baidu::xpu::api::Context* p; ~ContextGuard(){if(p)baidu::xpu::api::destroy_context(p);} } context_guard{ctx};
    Buffer scale(device,128*sizeof(float));
    std::vector<float> maxima(128,127.0f);
    X(xpu_memcpy(scale.p,maxima.data(),maxima.size()*sizeof(float),XPU_HOST_TO_DEVICE)); X(xpu_wait());
    if(dtype=="INT8") {
        if(!ctx) throw std::runtime_error("XBLAS context creation failed");
        std::cerr << "compute_api=XBLAS fc_fusion<int8_t,int8_t,float,int8_t,float,float>; FP32 output; maxima=127; alpha=1 beta=0; no bias; LINEAR activation\n";
    } else std::cerr << "compute_api=XBLAS cublasGemmEx FP32 accumulation/output\n";
    auto work=[&]() {
        if(dtype=="INT8") {
            X((baidu::xpu::xblas::fc_fusion<int8_t,int8_t,float,int8_t,float,float>(ctx,
                static_cast<int8_t*>(a.p),static_cast<int8_t*>(b.p),static_cast<float*>(c.p),
                n,n,n,false,true,static_cast<float*>(scale.p),static_cast<float*>(scale.p),nullptr,
                n,n,n,1.0f,0.0f,nullptr,baidu::xpu::api::Activation_t::LINEAR)));
            X(xpu_wait(ctx->xpu_stream));
        } else {
        X(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,
          &alpha,
          a.p,type,n,b.p,type,n,
          &beta,c.p,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        }
        X(xpu_wait());
    };
    samples.measure(work,warmup,count,seconds);
    std::vector<unsigned char> hc(elements*4);
    X(xpu_memcpy(hc.data(),c.p,hc.size(),XPU_DEVICE_TO_HOST)); X(xpu_wait());
    size_t checked= n<=128 ? elements : 257;
    for(size_t j=0;j<checked;j++) {
        size_t idx=n<=128 ? j : (j*104729+17)%elements;
        int r=idx%n, col=idx/n; double expected=0;
        for(int k=0;k<n;k++) expected+=dtype=="FP32" ? double(afloat(r,k))*double(bfloat(k,col)) : double(aval(r,k)*bval(k,col));
        double actual=reinterpret_cast<float*>(hc.data())[dtype=="INT8" ? size_t(r)*n+col : idx];
        double tolerance=dtype=="FP32" ? 0.0001 + std::abs(expected)*0.00001 : 0.01;
        if(!std::isfinite(actual) || std::abs(actual-expected)>tolerance)
            throw std::runtime_error("GEMM correctness failure at " + std::to_string(idx) + ": actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
    return checked;
}
int main(int argc,char** argv) {
    try {
        std::map<std::string,std::string> opt;
        for(int i=1;i<argc;i+=2) {
            if(i+1>=argc || std::string(argv[i]).rfind("--",0)!=0) throw std::runtime_error("expected --key value");
            opt[argv[i]]=argv[i+1];
        }
        auto get=[&](const std::string& k,const std::string& d){return opt.count(k) ? opt.at(k) : d;};
        auto mode=get("--mode","inventory");
        library(reinterpret_cast<const void*>(&xpu_memcpy));
        if(mode=="inventory") {
            int count=0; X(xpu_device_count(&count));
            if(count<1 || count>8) throw std::runtime_error("invalid selected native device count");
            // This compatibility build's xpu_device_list returns success but
            // leaves the caller's array unchanged. Enumerate local ordinals;
            // the caller must join their observed PCI addresses to host UUIDs.
            std::vector<int> ids(count);for(int i=0;i<count;i++)ids[i]=i;
            std::cout << "P800_INVENTORY [";
            for(int i=0;i<count;i++) {auto address=pci(ids[i]);std::cout << (i ? "," : "") << "{\"native_id\":" << ids[i] << ",\"pci_bdf\":\"" << address << "\"}";}
            std::cout << "]" << std::endl;return 0;
        }
        int src=std::stoi(opt.at("--src")), dst=std::stoi(get("--dst",std::to_string(src)));
        bind(src,opt.at("--src-pci"));
        if(mode=="p2p" || mode=="p2p-bidir") bind(dst,opt.at("--dst-pci"));
        X(xpu_set_device(src));
        size_t bytes=std::stoull(get("--bytes","1048576"));
        int n=std::stoi(get("--dimension","2048")), count=std::stoi(get("--samples","25")),warmup=std::stoi(get("--warmup","5"));
        double seconds=std::stod(get("--minimum-seconds","0"));
        if(bytes<1 || bytes>1073741824ULL || n<8 || n>8192 || n%8 || count<1 || count>100000 || warmup<0 || warmup>100 || seconds<0 || seconds>120)
            throw std::runtime_error("out-of-bounds workload arguments");
        Samples samples;size_t checked=0;
        if(mode=="gemm") {
            auto dtype=opt.at("--dtype");
            if(dtype!="FP32" && dtype!="FP16" && dtype!="BF16" && dtype!="INT8") throw std::runtime_error("unsupported dtype");
            checked=gemm(dtype,src,n,samples,warmup,count,seconds);
        } else if(mode=="d2d-kernel") {
            checked=kernel_copy(src,bytes,samples,warmup,count,seconds);
        } else if(mode=="p2p-bidir") {
            checked=bidirectional(src,dst,bytes,samples,warmup,count,seconds);
        } else if(mode=="h2d" || mode=="d2h" || mode=="d2d" || mode=="p2p") {
            auto api=get("--pinned-api","register");
            if(api!="register" && api!="alloc")throw std::runtime_error("invalid pinned API");
            size_t chunk=std::stoull(get("--async-chunk-bytes","1048576"));
            if(chunk>1073741824ULL)throw std::runtime_error("async chunk is out of bounds");
            checked=copy_bench(mode,src,dst,bytes,get("--pinned","0")=="1",get("--async","0")=="1",api=="register",chunk,samples,warmup,count,seconds);
        } else throw std::runtime_error("unsupported mode");
        samples.print(mode,bytes,n,warmup,checked,get("--dtype","bytes"));
        std::ifstream maps("/proc/self/maps");std::string line;
        while(std::getline(maps,line)) if(line.find(".so")!=std::string::npos && (line.find("xpu")!=std::string::npos || line.find("cublas")!=std::string::npos)) std::cerr << "loaded_map=" << line << '\n';
        return 0;
    } catch(const std::exception& e) {std::cerr << "P800_ERROR " << e.what() << std::endl; return 1;}
}
