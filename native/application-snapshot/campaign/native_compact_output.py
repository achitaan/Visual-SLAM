"""Lossless encoding AFTER the owned native writer exits; no numerical transform."""
from pathlib import Path
import gzip,hashlib,json,math,os,re,shutil
from native_local_output import NativeLocalOutput, signature, directory, FILE

SNAPSHOT=re.compile(r'snapshot-(0|[1-9][0-9]*)\.json')
CHUNK=1024*1024
MAX_SNAPSHOT_BYTES=512*1024*1024

def digest(path,budget=lambda:None):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(CHUNK),b''):budget();h.update(part)
    return h.hexdigest()

def check_roundtrip(raw,encoded,budget):
    h=hashlib.sha256();count=0
    with raw.open('rb') as source,gzip.open(encoded,'rb') as decoded:
        while True:
            budget();left=source.read(CHUNK);right=decoded.read(CHUNK)
            if left!=right:raise ValueError('Encoded snapshot changed a payload byte')
            if not left:break
            count+=len(left);h.update(left)
    return count,h.hexdigest()

class NativeCompactOutput(NativeLocalOutput):
    def export_compact(self,*,deadline,writer_exited,native_hashes):
        if not math.isfinite(deadline):raise ValueError('Nonfinite absolute deadline')
        if writer_exited is not True:raise ValueError('Cannot encode a live writer')
        if self.exported:raise ValueError('Export already completed')
        def budget():
            directory(self.stage);directory(self.destination)
            if (self.stage.stat().st_dev,self.stage.stat().st_ino)!=self.created_signature:
                raise ValueError('Stage replaced')
            if self.clock()>=deadline:raise TimeoutError('Absolute arm deadline during compact export')
            if shutil.disk_usage(self.physical_root).free<=self.reserve_bytes:raise OSError('Physical reserve reached')
        budget();started=self.clock();files=sorted(self.stage.iterdir(),key=lambda p:p.name)
        if set(native_hashes)!={p.name for p in files}:raise ValueError('Native file set changed')
        records={};converted=[]
        for source in files:
            if not FILE.fullmatch(source.name) or source.name in ('.','..'):raise ValueError('Unscoped filename')
            before=signature(source);expected=native_hashes[source.name]
            if digest(source,budget)!=expected:raise ValueError('Native bytes changed before export')
            encoded=bool(SNAPSHOT.fullmatch(source.name))
            if encoded and before[2]>MAX_SNAPSHOT_BYTES:raise ValueError('Snapshot exceeds declared decoder bound')
            output_name=source.name+'.gz' if encoded else source.name
            target=self.destination/output_name
            if target.exists() or target.is_symlink():raise FileExistsError(target)
            stored=source
            if encoded:
                stored=self.stage/output_name
                if stored.exists() or stored.is_symlink():raise FileExistsError(stored)
                with source.open('rb') as inp,stored.open('xb') as raw:
                    with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=1,mtime=0) as out:
                        for part in iter(lambda:inp.read(CHUNK),b''):budget();out.write(part)
                    raw.flush();os.fsync(raw.fileno())
            stored_before=signature(stored);stored_hash=digest(stored,budget)
            with stored.open('rb') as inp,target.open('xb') as out:
                if (os.fstat(inp.fileno()).st_dev,os.fstat(inp.fileno()).st_ino)!=stored_before[:2]:raise ValueError('Encoded source replaced')
                for part in iter(lambda:inp.read(CHUNK),b''):budget();out.write(part)
                out.flush();os.fsync(out.fileno())
            if signature(stored)!=stored_before or digest(target,budget)!=stored_hash:raise ValueError('Final encoded copy differs')
            if encoded:
                for path in (stored,target):
                    count,h=check_roundtrip(source,path,budget)
                    if count!=before[2] or h!=expected:raise ValueError('Roundtrip identity differs')
            else:
                if digest(target,budget)!=expected:raise ValueError('Plain output differs')
            if signature(source)!=before or digest(source,budget)!=expected:raise ValueError('Native source changed during export')
            records[source.name]=dict(output_name=output_name,encoding='gzip1' if encoded else 'plain',
                raw_bytes=before[2],raw_sha256=expected,bytes=stored_before[2],sha256=stored_hash)
            if encoded:
                # Convert only a NEW private intermediate after TWO exact copies.
                # Both encoded payloads remain; no previously completed evidence is touched.
                budget()
                if source.resolve().parent!=self.stage.resolve() or source.is_symlink() or signature(source)!=before:
                    raise ValueError('Private intermediate scope changed')
                source.unlink();converted.append(source.name)
        budget()
        if {p.name for p in self.stage.iterdir()}!={v['output_name'] for v in records.values()}:raise ValueError('Final stage set changed')
        self.exported=True
        return dict(status='verified_lossless_compact_export_two_payload_copies',stage=str(self.stage),destination=str(self.destination),
            files=records,raw_bytes=sum(v['raw_bytes'] for v in records.values()),bytes=sum(v['bytes'] for v in records.values()),
            wall_s=self.clock()-started,absolute_deadline_s=deadline,writer_exited=True,
            converted_new_private_intermediates=converted,unique_payloads_deleted=0,compression_level=1)

def snapshot_artifact(case,folder,frame):
    if type(frame) is not int or frame<0:raise ValueError('Invalid original frame ID')
    source=f'snapshot-{frame}.json';entry=case['output_transport']['files'][source]
    if entry['encoding']!='gzip1' or entry['output_name']!=source+'.gz':raise ValueError('Unscoped snapshot metadata')
    path=Path(folder)/entry['output_name'];directory(Path(folder));signature(path)
    if digest(path)!=entry['sha256'] or case['output_hashes'][path.name]!=entry['sha256']:raise ValueError('Encoded artifact changed')
    limit=entry['raw_bytes']
    if not isinstance(limit,int) or limit<0 or limit>MAX_SNAPSHOT_BYTES:raise ValueError('Invalid decoder bound')
    payload=bytearray();h=hashlib.sha256()
    with gzip.open(path,'rb') as stream:
        while True:
            part=stream.read(min(CHUNK,limit-len(payload)+1))
            if not part:break
            if len(payload)+len(part)>limit:raise ValueError('Decoded artifact exceeds bound')
            payload.extend(part);h.update(part)
    if len(payload)!=limit or h.hexdigest()!=entry['raw_sha256']:raise ValueError('Decoded payload changed')
    return path,json.loads(payload)
