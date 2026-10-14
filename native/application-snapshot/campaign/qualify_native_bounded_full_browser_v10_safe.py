"""One-shot normal/startup/mapping/finalization browser qualifications.

Only dispatch after source-specific seed timing and saved verifier are terminal
and review24 component tests pass. Each mode has one fresh scoped run; stop
first incomplete outcome. Functional checks do not replace accuracy evidence.
"""
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse

B = Path(__file__).resolve().parent
PINNED = Path(r'C:\Users\Achita\.codex\worktrees\e824\VSLAM\.venv-performance-gpu\Scripts\python.exe')
NODE = Path(r'C:\Program Files\nodejs\node.exe')
CLI = Path(r'C:\Users\Achita\AppData\Roaming\npm\node_modules\@playwright\cli\playwright-cli.js')
WSL_PY = '/root/slam-native-20261009/env/bin/python'


class FinalizerObservation:
    """Qualification-only one-probe reader; no control or estimator authority."""
    def __init__(self, run):
        self.run=run;self.proof=None;self.error=None;self.attempted=False
        observer=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                if self.path!='/finalizer?run='+run:
                    self.send_error(404);return
                status=200
                try:
                    if observer.error is not None:raise RuntimeError(observer.error)
                    if observer.proof is None:
                        path=B/(run+'-admissions-v3.jsonl')
                        rows=[]
                        if path.exists():
                            if path.stat().st_size>1024*1024:raise ValueError('bounded journal required')
                            rows=[json.loads(line) for line in path.read_text().splitlines()]
                        if any(row.get('role')=='raw_finalizer' and row.get('kind')=='owned_group' for row in rows):
                            if observer.attempted:raise RuntimeError('one actual phase probe only')
                            observer.attempted=True
                            probe='/mnt/c/'+str(B/'observe_native_release_finalizer.py')[3:].replace('\\','/')
                            reply=subprocess.run(['wsl.exe','-d','Ubuntu','-u','root','--',WSL_PY,'-B',probe,run],
                                capture_output=True,text=True,encoding='utf8',timeout=5)
                            if reply.returncode:raise RuntimeError('actual finalizer probe failed: '+reply.stderr)
                            observer.proof=json.loads(reply.stdout)
                    payload=dict(ready=observer.proof is not None,proof=observer.proof)
                except BaseException as error:
                    observer.error=repr(error);status=500;payload=dict(error=observer.error)
                raw=json.dumps(payload).encode('utf8')
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        self.server=HTTPServer(('127.0.0.1',56313),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)

    def start(self):self.thread.start()
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=6)
        if self.thread.is_alive():raise RuntimeError('qualification observer did not stop')


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(mode, sequence, frames, seed, suffix):
    if mode not in {'normal', 'mapping'} or not suffix.isalnum():
        raise ValueError('one declared mode and alphanumeric attempt identity required')
    if Path(sys.executable).resolve() != PINNED.resolve():
        raise ValueError('pinned GPU Python -B required')
    from native_application_presets_v2 import select_preset
    prepared = select_preset(sequence, frames, seed)
    cap = prepared['spec']['cap_s']
    from run_serialized_release_representative import no_other_worker
    no_other_worker()
    verification = json.loads((B / 'native-serialized-release-seed-confirmation-verification.json').read_text())
    if verification.get('status') != 'verified_complete_source_specific_seed_confirmation_not_retention':
        raise ValueError('all source seed blocks require saved verification before application qualification')
    # Required actual components, not the unexecuted implementation manifest.
    focused = json.loads((B / 'native-review24-focused-components-recovery-verification.json').read_text())
    if focused.get('status') != 'qualified_review24_focused_components_not_application_retention':
        raise ValueError('executed terminal/origin/guardian/normal-group components required')
    for name, expected in focused['successor_sha256'].items():
        if sha(B / name) != expected:
            raise ValueError('qualified component identity changed: ' + name)
    readiness=json.loads((B/'native-application-readiness-gate-local-verification.json').read_text())
    if readiness.get('status')!='qualified_initial_admission_read_order_not_application_acceptance':
        raise ValueError('executed readiness-gate component required')
    if readiness['successor_sha256']!=sha(B/'native_application_follower_v4.py'):
        raise ValueError('qualified readiness follower identity changed')
    run = f'native_prototype_fullappv10_{mode}_{sequence}_{frames}_seed{seed}_{suffix.lower()}'
    session = f'native-fullapp-v10-{mode}-{sequence}'
    prefix = B / f'native-fullapp-v10-{mode}-{sequence}-{frames}-seed{seed}-{suffix.lower()}'
    result_path = Path(str(prefix) + '-result.json')
    if result_path.exists() or (B / (run + '_application_release_v10')).exists():
        raise FileExistsError('each prepared qualification is one attempt only')

    def cli(*args, timeout=120):
        reply = subprocess.run([str(NODE), str(CLI), '-s=' + session, *args], cwd=B,
                               capture_output=True, text=True, encoding='utf8', timeout=timeout)
        with Path(str(prefix) + '-cli.log').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(list(args)) + '\n' + reply.stdout + reply.stderr + '\n')
        if reply.returncode:
            raise RuntimeError('owned browser CLI failure, original output preserved')
        return reply.stdout

    # A dedicated owned browser, never the user's existing tab or root server.
    cli('open', 'about:blank')
    command = [str(PINNED), '-B', str(B / 'native_application_launcher_release_v10.py'),
               '--run-id', run, '--sequence', sequence, '--frames', str(frames), '--seed', str(seed),
               '--start-timeout', '120', '--start-delay', '20', '--serve-after', '10',
               '--control-port', '56311', '--telemetry-port', '56312']
    result = dict(status='incomplete_no_pass', mode=mode, run_id=run, sequence=sequence, frames=frames, seed=seed, prepared_preset=prepared,
                  utc=datetime.now(timezone.utc).isoformat(), command=command,
                  launcher_sha256=sha(B / 'native_application_launcher_release_v10.py'),
                  browser_component_sha256=sha(B / 'NativeOwnedStopControlV2.tsx'),
                  ground_truth_read=False, estimator_accuracy_claim=False,
                  native_rate_claim=False, retained=False, main_changed=False)
    worker = subprocess.Popen(command, cwd=B, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf8', bufsize=1)
    result['launcher_pid'] = worker.pid
    phase_observer=None
    from native_browser_receipt_sink import ReceiptSink
    sink=ReceiptSink(Path(str(prefix)+'-browser-receipts.jsonl'),run)
    sink.start()
    from native_physical_disk_sample import PhysicalDiskSample
    disk=PhysicalDiskSample(B.parents[2],1610612736);disk.start()
    try:
        first = worker.stdout.readline()
        Path(str(prefix) + '-launcher.log').write_text(first, encoding='utf8')
        ready = json.loads(first)
        url = ready['browser_url']; parsed = urlparse(url); query = parse_qs(parsed.query)
        assert parsed.netloc == '127.0.0.1:4301' and parsed.path == '/native-stop-v2'
        assert query['run'] == [run] and query['control'] == ['ws://127.0.0.1:56311/']
        assert query['ws'] == ['ws://127.0.0.1:56312/']
        if mode=='finalization':
            phase_observer=FinalizerObservation(run);phase_observer.start()
        spec = dict(run=run, mode=mode, frames=frames, cap_s=cap, receipt_url=sink.url, max_pending=32, url=url, control=query['control'][0], telemetry=query['ws'][0],
                    phase_url='http://127.0.0.1:56313/finalizer?run='+run,
                    screenshot=str(prefix) + '-browser.png')
        code = r'''async page => {
          const spec=SPEC;
          const observation={run_id:spec.run,mode:spec.mode,page_errors:[],decode_errors:[],
            control_states:[],receipt_count:0,ignored_websocket_urls:[],stop_clicked:false};
          page.on('pageerror',e=>observation.page_errors.push(String(e)));
          let mapBeforeEnd=false,lastTelemetry=null,pending=0,recordingError=null;
          let receiptTail=Promise.resolve();
          const queueReceipt=value=>{
            if(recordingError)return;
            const payload=JSON.stringify(value);
            if(payload.length>2*1024*1024 || pending>=spec.max_pending) {
              recordingError='bounded receipt queue/body exceeded';return;
            }
            pending++;
            receiptTail=receiptTail.then(async()=>{
              const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),5000);
              try {
                const response=await fetch(spec.receipt_url,{method:'POST',
                  headers:{'Origin':'http://127.0.0.1:4301','Content-Type':'application/json'},
                  body:payload,signal:controller.signal});
                if(!response.ok)throw new Error('receipt sink status '+response.status);
                observation.receipt_count++;
              } finally {pending--;clearTimeout(timer);}
            }).catch(error=>{recordingError=String(error);});
          };
          page.on('websocket',ws=>{
            // Identify the exact bound application stream BEFORE any decoding.
            if(ws.url()!==spec.control && ws.url()!==spec.telemetry) {
              observation.ignored_websocket_urls.push(ws.url());return;
            }
            ws.on('framereceived',event=>{
              try {
                const value=JSON.parse(String(event.payload));
                if(value.run_id!==spec.run)throw new Error('wrong application run');
                if(ws.url()===spec.control) {
                  const {capability,...safe}=value;observation.control_states.push(safe);
                } else {
                  lastTelemetry=value;queueReceipt(value);
                  if(value.native_snapshot && value.frame_index+1<spec.frames)mapBeforeEnd=true;
                }
              } catch(e){observation.decode_errors.push({url:ws.url(),error:String(e)});}
            });
          });
          await page.goto(spec.url,{waitUntil:'domcontentloaded',timeout:30000});
          if(spec.mode!=='normal') {
            const end=Date.now()+(spec.cap_s+60)*1000;
            while(Date.now()<end) {
              const state=await page.locator('[aria-label="Native run control"]').innerText();
              const button=page.getByRole('button',{name:'Stop run',exact:true});
              let proof=null;
              if(spec.mode==='startup' && state.includes('Preparing native run') && !await button.isDisabled())
                proof={kind:'prepared_before_spawn'};
              if(spec.mode==='mapping' && mapBeforeEnd && state.includes('Native run active') && !await button.isDisabled())
                proof={kind:'actual_browser_map_before_last_frame',last_message:lastTelemetry};
              if(spec.mode==='finalization' && state.includes('Native run active') && !await button.isDisabled()) {
                const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),6000);
                try {
                  const reply=await fetch(spec.phase_url,{signal:controller.signal});
                  const phase=await reply.json();if(!reply.ok)throw new Error(phase.error);
                  if(phase.ready)proof=phase.proof;
                } finally {clearTimeout(timer);}
              }
              if(proof) {
                observation.stop_trigger=proof;
                observation.stop_click_epoch_ms=Date.now();
                await button.click({timeout:1000});observation.stop_clicked=true;break;
              }
              await new Promise(resolve=>setTimeout(resolve,10));
            }
            if(!observation.stop_clicked)throw new Error('declared phase not qualified; no retry');
          }
          const terminal=spec.mode==='normal'?'Run complete':'Cancelled \u00b7 cleanup confirmed \u00b7 incomplete';
          await page.waitForFunction(expected=>{
              const status=document.querySelector('[aria-label="Native run control"] [role="status"]')?.textContent;
              return status===expected || status==='Run failed \u00b7 incomplete';
            },
            terminal,{timeout:(spec.cap_s+60)*1000});
          if(await page.locator('[aria-label="Native run control"] [role="status"]').textContent()!==terminal)
            throw new Error('application explicitly failed; original run preserved, no replay');
          await receiptTail;
          if(recordingError || pending)throw new Error(recordingError || 'receipt queue incomplete');
          observation.receipt_queue_bound=spec.max_pending;
          observation.final_display=await page.locator('[aria-label="Native run control"]').innerText();
          observation.final_stop_disabled=await page.getByRole('button',{name:'Stop run',exact:true}).isDisabled();
          await page.screenshot({path:spec.screenshot,fullPage:true});return observation;
        }'''.replace('SPEC', json.dumps(spec))
        script = Path(str(prefix) + '-browser.js'); script.write_text(code, encoding='utf8')
        output = cli('run-code', '--filename=' + str(script), timeout=cap+150)
        payload = output.split('### Result\n', 1)[1].split('\n###', 1)[0].strip()
        observation = json.loads(payload); save(Path(str(prefix) + '-observation.json'), observation)
        sink.close();result['browser_receipts']=sink.summary()
        disk.close();result['physical_disk_samples']=disk.summary()
        assert result['physical_disk_samples']['sampled_reserve_gate_pass'],'Physical reserve gate failed; no acceptance'
        assert not result['browser_receipts']['error'] and result['browser_receipts']['records']==observation['receipt_count']
        rest = worker.communicate(timeout=45)[0]
        with Path(str(prefix) + '-launcher.log').open('a', encoding='utf8') as stream: stream.write(rest)
        result['launcher_exit_code'] = worker.returncode
        app_path = B / (run + '_application_release_v10/result.json')
        application = json.loads(app_path.read_text())
        assert worker.returncode == 0 and application['cleanup_confirmed']
        assert not observation['page_errors'] and not observation['decode_errors']
        assert observation['final_stop_disabled']
        ids = application['emitted_original_ids']
        assert ids == list(range(len(ids))) and application['unprocessed_original_ids'] == list(range(len(ids),frames))
        if mode == 'normal':
            assert len(ids) == frames and application['status'] == 'completed_application_pending_independent_acceptance'
            assert application['accepted_snapshot_original_ids'] == sorted(set(range(prepared['spec']['snapshot_interval'],frames,prepared['spec']['snapshot_interval'])) | {frames-1})
            assert application['completion_proof'] and application['stop_state']['phase'] == 'completed'
        elif mode == 'startup':
            assert application['status'] == 'cancelled_before_spawn_incomplete' and not ids
            assert application['process_created'] is False and not (B / run).exists()
        else:
            assert application['status'] == 'user_cancelled_incomplete_not_acceptance'
            assert application['stop_state']['phase'] == 'cancelled'
            if mode == 'mapping':
                assert 0 < len(ids) < frames and application['accepted_snapshot_original_ids']
            else:
                assert observation['stop_trigger']['status'] == 'actual_same_boot_admitted_raw_finalizer_alive_before_browser_stop'
        result.update(status='actual_declared_browser_phase_complete_pending_saved_verification',
                      emitted_frames=len(ids), application_result=str(app_path),
                      application_result_sha256=sha(app_path), cleanup_confirmed=True)
    except BaseException as error:
        result.update(status='incomplete_no_pass',error=repr(error),launcher_live_on_failure=worker.poll() is None)
        if worker.poll() is not None:
            try:
                rest=worker.communicate(timeout=5)[0]
                with Path(str(prefix)+'-launcher.log').open('a',encoding='utf8') as stream:stream.write(rest)
            except BaseException as log_error:
                result['launcher_log_collection_error']=repr(log_error)
        raise
    finally:
        # Cleanup attempts are independent; never reassert a known failed gate before saving it.
        result['qualification_cleanup_errors']={}
        for name,helper,summary_key in (
                ('physical_sampler',disk,'physical_disk_samples'),
                ('receipt_sink',sink,'browser_receipts')):
            try:
                helper.close()
                result[summary_key]=helper.summary()
                result[name+'_stopped']=True
            except BaseException as cleanup_error:
                result[name+'_stopped']=False
                result['qualification_cleanup_errors'][name]=repr(cleanup_error)
        samples=result.get('physical_disk_samples',{})
        if not samples.get('sampled_reserve_gate_pass',False):
            result['status']='incomplete_no_pass'
            result.setdefault('error','Physical reserve sample gate failed or unavailable; no acceptance')
        if phase_observer is not None:
            try:phase_observer.close();result['qualification_observer_stopped']=True
            except BaseException as cleanup_error:
                result['qualification_observer_stopped']=False
                result['qualification_cleanup_errors']['phase_observer']=repr(cleanup_error)
        try:
            cli('close',timeout=15)
            result['owned_browser_closed']=True
        except BaseException as cleanup_error:
            result['owned_browser_closed']=False
            result['qualification_cleanup_errors']['owned_browser']=repr(cleanup_error)
        if result['qualification_cleanup_errors']:
            result['status']='incomplete_no_pass'
            result.setdefault('error','Qualification helper cleanup unconfirmed; no acceptance')
        try:
            save(result_path,result)
        except BaseException as persistence_error:
            # A real filesystem failure cannot be promised away; preserve the primary exception.
            result['status']='incomplete_no_pass'
            result['failure_record_persistence_error']=repr(persistence_error)
            sys.stderr.write(json.dumps(result,allow_nan=False)+'\n')
    if result['status']=='incomplete_no_pass':
        raise RuntimeError(result.get('error','Qualification incomplete; no acceptance'))
    print(json.dumps({k:result.get(k) for k in ("status","mode","run_id","emitted_frames","cleanup_confirmed")}))


if __name__ == '__main__':
    if len(sys.argv)!=6:raise ValueError('MODE SEQUENCE FRAMES SEED SUFFIX required; one prepared attempt')
    qualify(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])
