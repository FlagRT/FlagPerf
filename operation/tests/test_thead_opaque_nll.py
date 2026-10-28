"""Opaque PPU C++ NLL requires exact records, kernels and backward linkage."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vendors.thead.dispatch import analyze_dispatch


class OpaqueNllTests(unittest.TestCase):
    def fixture(self):
        events = [
            dict(ph='X', cat='user_annotation', name='ppu_target_first_0', pid=1, tid=2, ts=0, dur=200),
            dict(ph='X', cat='user_annotation', name='ppu_dispatch_0', pid=1, tid=2, ts=10, dur=40),
            dict(ph='X', cat='cpu_op', name='aten::nll_loss_forward', pid=1, tid=2, ts=69, dur=22,
                 args={'Input Dims': [[8, 1024], [8], [], [], []]}),
            dict(ph='X', cat='cpu_op', name='NllLossForwardFunction', pid=1, tid=2, ts=70, dur=20,
                 args={'External id': 200, 'Sequence number': 3, 'Input type': ['float', 'long int', '', 'Scalar', 'Scalar'],
                       'Input Dims': [[8, 1024], [8], [], [], []]}),
            dict(ph='X', cat='cpu_op', name='NllLossForwardFunctionBackward', pid=1, tid=3, ts=120, dur=20,
                 args={'External id': 300, 'Sequence number': 3}),
            dict(ph='s', cat='fwdbwd', id=500, pid=1, tid=2, ts=70),
            dict(ph='f', cat='fwdbwd', id=500, pid=1, tid=3, ts=120),
        ]
        for corr, host_ts, host_tid, external, kernel in (
                (7, 20, 2, 100, 'log_softmax_kernel'),
                (8, 80, 2, 200, 'nll_loss_forward_kernel'),
                (9, 130, 3, 300, 'nll_loss_backward_kernel')):
            events.extend([
                dict(ph='X', cat='cuda_driver', name='cuLaunchKernelEx', pid=1, tid=host_tid,
                     ts=host_ts, dur=1, args={'correlation': corr, 'External id': external}),
                dict(ph='s', cat='ac2g', id=corr, pid=1, tid=host_tid, ts=host_ts),
                dict(ph='X', cat='kernel', name=kernel, pid=4, tid=5, ts=corr*100, dur=3,
                     args={'correlation': corr, 'device': 0, 'stream': 1}),
                dict(ph='f', cat='ac2g', id=corr, pid=4, tid=5, ts=corr*100)])
        rows = [dict(index=0, op='aten::_log_softmax', device_input=True, device_output=True, cpu_transfer=False)]
        return events, rows

    def check(self, events, rows, **kwargs):
        return analyze_dispatch(events, rows, case='cross_entropy_loss', dtype='FP32', **kwargs)

    def test_reviewed_cpp_forward_backward_complete_composite_route(self):
        for dtype, scalar in (('FP32', 'float'), ('FP16', 'c10::Half'), ('BF16', 'c10::BFloat16')):
            events, rows = self.fixture()
            events[3]['args']['Input type'][0] = scalar
            result = analyze_dispatch(events, rows, case='cross_entropy_loss', dtype=dtype)
            self.assertEqual(result['status'], 'passed', result)
            self.assertEqual([c['op'] for c in result['computations']],
                             ['aten::_log_softmax', 'NllLossForwardFunction', 'NllLossForwardFunctionBackward'])

    def test_cpp_evidence_does_not_qualify_other_cases_or_unknown_dtypes(self):
        events, rows = self.fixture()
        for case, dtype in (('abs', 'FP32'), ('cross_entropy_loss', 'INT32')):
            self.assertEqual(analyze_dispatch(events, rows, case=case, dtype=dtype)['status'], 'partial')

    def test_missing_ambiguous_or_mismatched_nll_proof_is_rejected(self):
        def bad_shape(events): events[3]['args']['Input Dims'][1] = [7]
        def bad_type(events): events[3]['args']['Input type'][0] = 'double'
        def wrong_sequence(events): events[4]['args']['Sequence number'] = 4
        def duplicate_op(events): events.append(copy.deepcopy(events[3]))
        def duplicate_edge(events): events.append(copy.deepcopy(events[5]))
        def no_edge(events): del events[5]
        def wrong_edge(events): events[6]['ts'] = 121
        def wrong_external(events): events[11]['args']['External id'] = 201
        def unknown_kernel(events): events[13]['name'] = 'unrelated_kernel'
        def missing_kernel(events): del events[13]
        def wrong_parent(events): events[2]['name'] = 'aten::unrelated'
        def wrong_cpp_record(events): events[3]['name'] = 'OpaqueFunction'
        def wrong_thread(events): events[11]['tid'] = 99
        def outside_record(events): events[3]['dur'] = 5
        for mutate in (bad_shape, bad_type, wrong_sequence, duplicate_op, duplicate_edge, no_edge,
                       wrong_edge, wrong_external, unknown_kernel, missing_kernel, wrong_parent,
                       wrong_cpp_record, wrong_thread, outside_record):
            with self.subTest(mutate=mutate.__name__):
                events, rows = self.fixture()
                mutate(events)
                self.assertEqual(self.check(events, rows)['status'], 'partial')

    def test_zero_initialization_alone_is_never_computation(self):
        events, rows = self.fixture()
        events = [events[0], events[1], *events[7:11]]
        rows[0].update(op='aten::zeros', device_input=False)
        events[4]['name'] = 'zeros_kernel'
        self.assertEqual(self.check(events, rows)['status'], 'partial')


if __name__ == '__main__':
    unittest.main()
