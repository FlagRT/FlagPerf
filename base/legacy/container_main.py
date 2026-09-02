# Copyright (c) 2024 BAAI. All rights reserved.
# Legacy dual-domain container entry retained during the Base CLI migration.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time
from loguru import logger
import os
import sys
from argparse import ArgumentParser
import subprocess


def resolve_benchmark_entrypoint(perf_path, case_spec, vendor):
    """Prefer a vendor Case entrypoint while preserving the generic fallback."""
    if ':' in case_spec:
        case_name, chip_model = case_spec.split(':', 1)
        vendor_selector = vendor + '/' + chip_model
        vendor_case_dir = os.path.join(
            perf_path, "benchmarks", case_name, vendor, chip_model
        )
    else:
        case_name = case_spec
        direct_vendor_dir = os.path.join(
            perf_path, "benchmarks", case_name, vendor
        )
        if os.path.isfile(os.path.join(direct_vendor_dir, "case_config.yaml")):
            vendor_selector = vendor
            vendor_case_dir = direct_vendor_dir
        else:
            chip_model = 'A100'
            vendor_selector = vendor + '/' + chip_model
            vendor_case_dir = os.path.join(direct_vendor_dir, chip_model)

    case_dir = os.path.join(perf_path, "benchmarks", case_name)
    vendor_main = os.path.join(vendor_case_dir, "main.py")
    entrypoint = vendor_main if os.path.isfile(vendor_main) else os.path.join(
        case_dir, "main.py"
    )
    return case_name, case_dir, entrypoint, vendor_selector


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--case_name",
                        type=str,
                        required=True,
                        help="case name")

    parser.add_argument("--nnodes",
                        type=int,
                        required=True,
                        help="number of node")

    parser.add_argument("--nproc_per_node",
                        type=int,
                        required=True,
                        help="*pu per node")

    parser.add_argument("--log_dir",
                        type=str,
                        required=True,
                        help="abs log dir")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")

    parser.add_argument("--log_level",
                        type=str,
                        required=True,
                        help="log level")

    parser.add_argument("--master_port",
                        type=int,
                        required=True,
                        help="master port")

    parser.add_argument("--master_addr",
                        type=str,
                        required=True,
                        help="master ip")

    parser.add_argument("--host_addr",
                        type=str,
                        required=True,
                        help="my ip")

    parser.add_argument("--node_rank",
                        type=int,
                        required=True,
                        help="my rank")

    parser.add_argument("--bench_or_tool",
                        type=str,
                        required=True,
                        help="benchmarks or toolkits")

    parser.add_argument("--perf_path",
                        type=str,
                        required=True,
                        help="abs path for FlagPerf/base")

    parser.add_argument(
        "--allow_disruptive_toolkit", action="store_true",
        help="allow evidence-first Ascend DMI performance commands",
    )

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def write_pid_file(pid_file_path, pid_file):
    '''Write pid file for watching the process later.
       In each round of case, we will write the current pid in the same path.
    '''
    pid_file_path = os.path.join(pid_file_path, pid_file)
    if os.path.exists(pid_file_path):
        os.remove(pid_file_path)
    file_d = open(pid_file_path, "w")
    file_d.write("%s\n" % os.getpid())
    file_d.close()


if __name__ == "__main__":
    config = parse_args()

    logfile = os.path.join(config.log_dir, config.case_name, config.host_addr + "_noderank" + str(config.node_rank), "container_main.log.txt")
    logger.remove()
    logger.add(logfile, level=config.log_level)
    logger.add(sys.stdout, level=config.log_level)

    logger.info(config)
    write_pid_file(config.log_dir, "start_base_task.pid")
    logger.info("Success Writing PID file at " + os.path.join(config.log_dir, "start_base_task.pid"))
    if config.bench_or_tool == "BENCHMARK":
        case_name, case_dir, entrypoint, vendor_selector = (
            resolve_benchmark_entrypoint(
                config.perf_path, config.case_name, config.vendor
            )
        )
        logger.info("Using PyTorch to Test {}'s {}".format(config.vendor, case_name))
        start_cmd = "cd " + case_dir + ";torchrun"
        # for torch
        start_cmd += " --nproc_per_node=" + str(config.nproc_per_node)
        start_cmd += " --nnodes=" + str(config.nnodes)
        start_cmd += " --node_rank=" + str(config.node_rank)
        start_cmd += " --master_addr=" + str(config.master_addr)
        start_cmd += " --master_port=" + str(config.master_port)
        # for flagperf
        start_cmd += " " + entrypoint
        start_cmd += " --vendor=" + vendor_selector
        start_cmd += " --node_size=" + str(config.nproc_per_node)
        script_log_file = os.path.join(os.path.dirname(logfile), "benchmark.log.txt")
    elif config.bench_or_tool == "TOOLKIT":
        if ':' not in config.case_name:
            case_name = config.case_name
            chip_model = 'A100'
        else:
            case_name, chip_model = config.case_name.split(':')
        logger.info("Using {}'s Toolkits to Test {}".format(config.vendor, case_name))
        case_dir = os.path.join(config.perf_path, "toolkits", case_name, config.vendor, chip_model)
        evidence_dir = os.path.join(os.path.dirname(logfile), "toolkit-evidence")
        start_cmd = "export NODERANK=" + str(config.node_rank) + ";"
        start_cmd += "export FLAGPERF_TOOLKIT_ARTIFACT_DIR=" + evidence_dir + ";"
        if config.allow_disruptive_toolkit:
            start_cmd += "export FLAGPERF_ALLOW_DISRUPTIVE_DMI=1;"
        start_cmd += "cd " + case_dir + ";bash main.sh"
        script_log_file = os.path.join(os.path.dirname(logfile), "toolkit.log.txt")
    else:
        logger.error("Invalid BENCHMARKS_OR_TOOLKITS CONFIG, STOPPED TEST!")
        exit(1)

    logger.info(start_cmd)
    logger.info(script_log_file)

    f = open(script_log_file, "w")
    p = subprocess.Popen(start_cmd,
                         shell=True,
                         stdout=f,
                         stderr=subprocess.STDOUT)
    p.wait()
    f.close()
    logger.info("Task Finish, return code: {}", p.returncode)
    sys.exit(p.returncode)
