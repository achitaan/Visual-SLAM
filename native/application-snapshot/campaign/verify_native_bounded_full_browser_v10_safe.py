"""Verify saved full UI bytes with bounded JSON/file reads; never replay native."""
import base64
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
from native_saved_observation_stream import ObservationStream

B=Path(__file__).resolve().parent
def load(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def verify(mode,sequence,frames,seed,suffix):
    if mode not in ('normal','mapping') or not suffix.isalnum():raise ValueError('exact declared run required')
    prefix=B/f'native-fullapp-v10-{mode}-{sequence}-{frames}-seed{seed}-{suffix.lower()}'
    run=f'native_prototype_fullappv10_{mode}_{sequence}_{frames}_seed{seed}_{suffix.lower()}'
    harness=load(Path(str(prefix)+'-result.json'))
    app=B/(run+'_application_release_v10');result=load(app/'result.json')
    protocol=load(B/(run.replace('_','-')+'-application-release-v10.json'))
    captured=load(Path(str(prefix)+'-saved-stage.json'))
    assert harness['status']=='actual_declared_browser_phase_complete_pending_saved_verification'
    assert harness['launcher_exit_code']==0 and harness['run_id']==run
    assert harness['owned_browser_closed'] and not harness['qualification_cleanup_errors']
    assert harness['physical_sampler_stopped'] and harness['receipt_sink_stopped']
    assert harness['application_result_sha256']==sha(app/'result.json')
    assert result['expected_frames']==frames and result['cleanup_confirmed']
    assert harness['launcher_sha256']==protocol['launcher_sha256']==sha(B/'native_application_launcher_release_v10.py')
    for name,digest in protocol['application_components_sha256'].items():assert sha(B/name)==digest,name
    assert protocol['presets_catalog_sha256']==sha(B/'native-application-prepared-presets.json')
    assert protocol['presets_selector_sha256']==sha(B/'native_application_presets_v2.py')
    ids=result['emitted_original_ids'];assert ids==list(range(len(ids)))
    assert result['unprocessed_original_ids']==list(range(len(ids),frames))
    assert captured['run_id']==run and captured['status']=='saved_terminal_native_stage_captured_no_replay'
    poses=(app/'poses.txt').read_bytes()
    native=base64.b64decode(captured['files']['online_poses.txt']['payload_base64'],validate=True)
    assert native.startswith(poses) and result['emitted_prefix_sha256']==sha(app/'poses.txt')
    trajectory=np.loadtxt(io.BytesIO(poses),ndmin=2).reshape(-1,3,4)[:,:,3].tolist()
    assert len(trajectory)==len(ids)
    for name,item in result['partial_artifacts'].items():
        assert captured['files'][name]['sha256']==item['sha256'],name
        assert captured['files'][name]['bytes']==item['bytes'],name
    observation={};displayed=set();receipt_count=0;map_before_end=False
    metadata_path=Path(str(prefix)+'-observation.json')
    assert metadata_path.stat().st_size<=2*1024*1024,'CLI returned unbounded observation metadata'
    observation=load(metadata_path)
    receipts=Path(str(prefix)+'-browser-receipts.jsonl')
    assert harness['browser_receipts']['sha256']==sha(receipts)
    assert not harness['browser_receipts']['error'] and harness['browser_receipts']['thread_stopped']
    assert observation['receipt_queue_bound']==32
    assert harness['physical_disk_samples']['sampled_reserve_gate_pass']
    max_receipt_bytes=0
    with receipts.open(encoding='utf8') as stream, (app/'published.jsonl').open(encoding='utf8') as ledger:
        for raw in stream:
            assert len(raw.encode('utf8'))<=2*1024*1024+1,'receipt exceeds declared bound'
            max_receipt_bytes=max(max_receipt_bytes,len(raw.encode('utf8')))
            value=json.loads(raw)
            line=ledger.readline()
            assert line and value==json.loads(line),'received FIFO message differs from immutable publication'
            assert value['run_id']==run
            index=value['frame_index'];assert index in ids
            assert value['trajectory']==trajectory[:index+1],'future or corrected emitted pose in browser'
            if value['native_snapshot']:
                assert value['native_snapshot']['original_id']<=index
                map_before_end |= index+1<frames
            displayed.add(index);receipt_count+=1
        assert ledger.readline()=='','browser skipped final publications'
    assert observation['run_id']==run and not observation['page_errors'] and not observation['decode_errors']
    assert receipt_count==observation['receipt_count']==harness['browser_receipts']['records']
    assert observation['final_stop_disabled'] and displayed==set(ids)
    controls=observation['control_states'];assert controls[0]['type']=='control_hello'
    assert controls[0]['phase']=='prepared' and all(c['run_id']==run for c in controls)
    audit_path=Path(result['whole_exit_audit']);audit=load(audit_path)
    assert result['whole_exit_audit_sha256']==sha(audit_path)
    assert audit['run_id']==run and audit['all_owned_exited'] and audit['seal']['all_owned_exited']
    for key in ('live_caller_observers','live_guardians_or_helpers','token_processes_outside_admissions'):
        assert not audit['proof'][key]
    assert all(not g['live_members'] for g in audit['proof']['owned_groups'])
    journal=(B/(run+'-admissions-v3.jsonl')).read_bytes();lines=journal.splitlines(keepends=True)
    rows=[json.loads(line) for line in lines]
    assert rows[-1]['event']=='journal_sealed' and rows[-1]==audit['seal']
    assert hashlib.sha256(b''.join(lines[:-1])).hexdigest()==audit['seal']['preseal_sha256']
    assert len([r for r in rows if r.get('event')=='process_admitted'])==audit['admission_count']
    assert all(r['run_id']==run and r['boot_id']==audit['boot_id'] for r in rows)
    assert audit['seal']['monotonic_s']<audit['seal']['original_deadline_monotonic_s']
    if mode=='normal':
        assert result['status']=='completed_application_pending_independent_acceptance' and len(ids)==frames
        interval=next(iter(protocol['runs'].values()))['snapshot_interval']
        assert result['accepted_snapshot_original_ids']==sorted(set(range(interval,frames,interval))|{frames-1})
        assert audit['gate_purpose']=='completed' and any(r['event']=='completion_gate_closed' for r in rows)
        assert not any(r['event']=='stop_gate_closed' for r in rows)
        assert not observation['stop_clicked'] and controls[-1]['phase']=='completed'
        assert controls[-1]['cleanup_confirmed'] and not controls[-1]['incomplete']
        assert result['completion_proof'] and (app/'map').is_dir() and map_before_end
    else:
        assert observation['stop_clicked'] and result['status']=='user_cancelled_incomplete_not_acceptance'
        assert controls[-1]['phase']=='cancelled' and controls[-1]['incomplete']
        assert any(c['phase']=='stopping' and not c['cleanup_confirmed'] for c in controls)
        assert 0<len(ids)<frames and result['accepted_snapshot_original_ids'] and map_before_end
        assert result['causal_observation']['positive_prefix_seen_while_linux_writer_alive']
    failure=B/'native-full01-application-resource-failure.json'
    resource_failed=failure.exists() and load(failure)['run_id']==run
    names=[Path(str(prefix)+suffix) for suffix in ('-result.json','-observation.json','-browser.png','-saved-stage.json')]
    names.append(receipts)
    names += [app/'result.json',B/(run.replace('_','-')+'-application-release-v10.json'),audit_path,B/(run+'-admissions-v3.jsonl')]
    if resource_failed:names.append(failure)
    record=dict(status=('verified_saved_functional_browser_delivery_storage_gate_failed_no_retention' if resource_failed else
        'verified_saved_bounded_full_browser_delivery_not_new_accuracy_or_retention'),
        utc=datetime.now(timezone.utc).isoformat(),run_id=run,sequence=sequence,frames=frames,seed=seed,
        original_id_coverage_preserved=True,emitted_frames=len(ids),receipt_count=receipt_count,
        strict_maps=len(result['accepted_snapshot_original_ids']),whole_owned_cleanup_confirmed=True,
        processing_and_cleanup_byte_consistency_pass=True,storage_gate_pass=not resource_failed,
        maximum_receipt_bytes=max_receipt_bytes,receipt_body_bound_bytes=2*1024*1024,physical_disk_samples=harness['physical_disk_samples'],
        source_byte_hashes={str(path.relative_to(B)):sha(path) for path in names},
        streaming_verifier_sha256=sha(Path(__file__)),parser_sha256=sha(B/'native_saved_observation_stream.py'),
        independent_review=False,estimator_replayed=False,ground_truth_read=False,
        new_accuracy_pair=False,native_rate_claim=False,retained=False,main_changed=False)
    out=Path(str(prefix)+'-streaming-verification.json')
    with out.open('x',encoding='utf8') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps({k:record[k] for k in ('status','emitted_frames','receipt_count','strict_maps','maximum_receipt_bytes','storage_gate_pass')}))

if __name__=='__main__':verify(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])
