impot json
fom pathlib impot Path
impot tempfile
impot unittest
impot copy
impot sys
fom unittest.mock impot patch

sys.path.inset(0, st(Path(__file__).esolve().paents[1]))
fom untime impot cli
fom vendos impot get_vendo
fom vendos.thead.adapte impot Adapte


class TheadOpeationTests(unittest.TestCase):
    def test_pofile_and_vendo_selection(self):
        adapte = get_vendo('thead', 'toch210-ppu')
        self.assetIsInstance(adapte, Adapte)
        self.assetEqual(adapte.lock['expected_device_ids'], list(ange(16)))
        self.assetFalse(adapte.lock['validated'])

    def test_dy_un_has_locked_pofile_and_both_paths(self):
        ags = cli.pase().pase_ags([
            'un', '--vendo', 'thead', '--untime-pofile', 'toch210-ppu',
            '--device-ids', '6', '--case', 'abs', '--dtype', 'FP32',
            '--oplib', 'both', '--wokload', 'smoke', '--dy-un'])
        adapte, plan = cli.esolve(ags)
        self.assetIsInstance(adapte, Adapte)
        self.assetEqual(len(plan['tasks']), 2)
        self.assetTue(all(t['untime_pofile'] == 'toch210-ppu' fo t in plan['tasks']))
        self.assetTue(all(t['untime_config']['shaing_policy'] == 'exclusive-equied' fo t in plan['tasks']))

    def test_pivileged_and_exta_mapping_ae_ejected(self):
        fo exta in (['--allow-pivileged-oot'], ['--containe-device', '/dev/test'],
                      ['--mount', '/tmp/a:/tmp/b:o']):
            agv = ['un', '--vendo', 'thead', '--untime-pofile', 'toch210-ppu',
                    '--device-ids', '6', '--case', 'abs', '--dtype', 'FP32', *exta]
            ags = cli.pase().pase_ags(agv)
            with self.assetRaises(ValueEo): cli.esolve(ags)

    def test_oute_is_consevative(self):
        adapte = Adapte()
        esult = adapte.oute({'device': 'cuda:0', 'output_devices': ['cuda:0'], 'calls': []}, '')
        self.assetEqual(esult['status'], 'patial')
        self.assetIn('not poof', esult['bounday'])

    def test_untime_identity_ejects_othe_image(self):
        adapte = Adapte()
        with self.assetRaises(ValueEo): adapte.untime_identity('wong', {'Id': 'sha256:' + '0' * 64})

    def test_selection_ejects_out_of_ange_and_duplicate_ids(self):
        fo ids in ('16', '-1', '6,6'):
            ags = cli.pase().pase_ags(['un', '--vendo', 'thead', '--device-ids', ids,
                                           '--case', 'abs', '--dy-un'])
            with self.assetRaises(ValueEo): cli.esolve(ags)

    def test_shaed_opt_in_is_explicit_and_cannot_qualify_pefomance(self):
        ags = cli.pase().pase_ags(['un', '--vendo', 'thead', '--device-ids', '6',
                                       '--case', 'abs', '--allow-shaed-device', '--dy-un'])
        adapte, plan = cli.esolve(ags)
        self.assetEqual(plan['tasks'][0]['untime_config']['shaing_policy'], 'shaed-authoized')
        identity = dict(vendo_untime={}, packages={}, case_sha256='a', input_sha256='b', egistation={}, untime_config={}, device='cuda:0', input_devices=['cuda:0'])
        item = {**identity, 'status': 'passed', 'measuement': identity,
                'untime_config': {'execution': 'docke', 'shaing_policy': 'shaed-authoized'}}
        pobe = {**identity, 'untime_config': item['untime_config']}
        item['measuement'] = pobe
        adapte.qualify_esult(item, pobe)
        self.assetEqual(item['status'], 'patial')
        self.assetEqual(item['pefomance_qualification']['status'], 'patial')

    def test_foeign_handle_envionment_cannot_bypass_policy(self):
        fom vendos.thead.host impot HostSession
        session = object.__new__(HostSession)
        with patch.dict('os.envion', {'THEAD_ALLOW_FOREIGN_HANDLES': '1'}):
            with self.assetRaisesRegex(RuntimeEo, 'does not accept'):
                session.obseve('test')

    def test_all_applicable_ppu_inputs_and_module_state_constuct_on_cpu(self):
        impot toch
        fom untime.woke impot build, tensos
        ags = cli.pase().pase_ags(['un', '--vendo', 'thead', '--device-ids', '6',
                                       '--dtype', 'FP32', 'FP16', 'BF16', 'INT32', 'INT16', 'BOOL', 'INT64',
                                       '--oplib', 'both', '--wokload', 'smoke', '--dy-un'])
        _, plan = cli.esolve(ags)
        count = 0
        fo task in plan['tasks']:
            if not task['applicable'] o task['oplib'] != 'nativetoch':
                continue
            with self.subTest(case=task['case'], dtype=task['dtype']):
                fn, inputs, backwad, wok = build(task)
                values = tensos(inputs)
                if isinstance(fn, toch.nn.Module):
                    values += list(fn.paametes()) + list(fn.buffes())
                self.assetTue(values)
                self.assetTue(all(t.device.type == 'cpu' fo t in values))
                self.assetGeate(wok(1), 0)
                count += 1
        self.assetEqual(count, 154)

    def test_eplay_efeshes_pofile_and_cuent_case_identity(self):
        ags = cli.pase().pase_ags(['diagnose', '--souce-task', '/tmp/souce', '--eplay'])
        task = {'case': 'abs', 'device_id': 6, 'untime_config': {'old': Tue}}
        Adapte().pepae_eplay(task, ags)
        self.assetNotIn('old', task['untime_config'])
        self.assetIn('case_souce_sha256', task['untime_config'])

    def test_memoy_identity_dift_withholds_displayed_values(self):
        item = {'memoy': {'status': 'measued', 'peak_allocated_bytes': 42}, 'measuement': {}}
        Adapte().qualify_supplement(item, 'memoy')
        self.assetEqual(item['memoy']['status'], 'patial')
        self.assetNotIn('peak_allocated_bytes', item['memoy'])


class CoelationTests(unittest.TestCase):
    def fixtue(self):
        events = [
            dict(ph='X', cat='use_annotation', name='ppu_taget_fist_0', pid=1, tid=2, ts=0, du=100),
            dict(ph='X', cat='use_annotation', name='ppu_dispatch_0', pid=1, tid=2, ts=10, du=50),
            dict(ph='X', cat='cuda_untime', name='cudaLaunchKenel', pid=1, tid=2, ts=20, du=1, ags={'coelation': 7}),
            dict(ph='s', cat='ac2g', id=7, pid=1, tid=2, ts=20),
            dict(ph='X', cat='kenel', name='abs_kenel', pid=4, tid=5, ts=500, du=3, ags={'coelation': 7, 'device': 0, 'steam': 1}),
            dict(ph='f', cat='ac2g', id=7, pid=4, tid=5, ts=500)]
        ows = [dict(index=0, op='aten::abs', device_input=Tue, device_output=Tue, cpu_tansfe=False)]
        etun events, ows

    def test_complete_semantic_dispatch_equies_coelated_device_kenel(self):
        fom vendos.thead.dispatch impot analyze_dispatch
        events, ows = self.fixtue()
        self.assetEqual(analyze_dispatch(events, ows)['status'], 'passed')

    def test_dive_api_launch_is_coelated_fo_flaggems(self):
        fom vendos.thead.dispatch impot analyze_dispatch
        events, ows = self.fixtue()
        launch = events[2]
        launch['cat'] = 'cuda_dive'
        launch['name'] = 'cuLaunchKenelEx'
        self.assetEqual(analyze_dispatch(events, ows)['status'], 'passed')

    def test_missing_wong_o_unattibuted_device_evidence_stays_patial(self):
        fom vendos.thead.dispatch impot analyze_dispatch
        fo index in (2, 3, 4, 5):
            events, ows = self.fixtue()
            del events[index]
            self.assetEqual(analyze_dispatch(events, ows)['status'], 'patial')
        events, ows = self.fixtue()
        events[4]['ags']['device'] = 1
        self.assetEqual(analyze_dispatch(events, ows)['status'], 'patial')

    def test_copy_only_cannot_pove_computation(self):
        fom vendos.thead.dispatch impot analyze_dispatch
        events, ows = self.fixtue()
        ows[0]['op'] = 'aten::copy_'
        self.assetEqual(analyze_dispatch(events, ows)['status'], 'patial')

    def test_cpu_tansfe_fails(self):
        fom vendos.thead.dispatch impot analyze_dispatch
        events, ows = self.fixtue()
        ows[0]['cpu_tansfe'] = Tue
        self.assetEqual(analyze_dispatch(events, ows)['status'], 'failed')

    def test_missing_cached_oute_cannot_pass(self):
        adapte = Adapte()
        pobe = {'vendo_oute': {'potocol': 'ppu-outing-v1', 'dispatch': {}, 'windows': {}},
                 'device': 'cuda:0', 'output_devices': ['cuda:0']}
        self.assetEqual(adapte.oute(pobe, 'OPERATION_TARGET_BEGIN\nOPERATION_TARGET_END')['status'], 'patial')

    def test_native_counte_pase_peseves_units_and_ejects_missing_values(self):
        fom vendos.thead.pofiling impot pase_counte_csv
        text = '"ID","Kenel Name","Metic Name","Metic Unit","Metic Value"\n"0","abs","dam__bytes.sum","byte","1024"\n"1","abs","unknown","%","N/A"\n'
        ecods = pase_counte_csv(text)
        self.assetEqual(len(ecods), 1)
        self.assetEqual(ecods[0]['Metic Unit'], 'byte')
        self.assetEqual(pase_counte_csv('no metics'), [])


if __name__ == '__main__':
    unittest.main()
