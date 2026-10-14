"""Owned Linux-local native output; hash-exact final export, no deletion or retries."""
from pathlib import Path
import hashlib, os, re, shutil, stat, time

STAGE_ROOT=Path('/root/slam-native-20261009/output-local-qualified')
NAME=re.compile(r'[a-z0-9_]+')
FILE=re.compile(r'[A-Za-z0-9_.-]+')

def signature(path):
    value=path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink!=1:
        raise ValueError('Output must be an owned regular single-link file')
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)

def directory(path):
    for p in (path,*path.parents):
        if p.exists() and (p.is_symlink() or not p.is_dir()):
            raise ValueError('Output directory has a symlink/non-directory ancestor')

class NativeLocalOutput:
    def __init__(self,destination,name,*,physical_root,reserve_bytes,stage_root=STAGE_ROOT,clock=time.monotonic):
        if not NAME.fullmatch(name):raise ValueError('Unscoped stage name')
        self.destination=Path(destination);self.physical_root=Path(physical_root)
        self.reserve_bytes=reserve_bytes;self.clock=clock
        if reserve_bytes<1610612736:raise ValueError('Reserve may not be lowered')
        directory(self.destination);directory(Path(stage_root))
        if not self.destination.is_dir():raise ValueError('Destination must already be exclusively created')
        Path(stage_root).mkdir(parents=True,exist_ok=True)
        self.stage=Path(stage_root)/name
        self.stage.mkdir(exist_ok=False)
        self.created_signature=self.stage.stat().st_dev,self.stage.stat().st_ino
        self.exported=False

    def export(self,*,deadline,writer_exited):
        if writer_exited is not True:raise ValueError('Cannot export a live writer')
        if self.exported:raise ValueError('Export already completed')
        directory(self.stage);directory(self.destination)
        if (self.stage.stat().st_dev,self.stage.stat().st_ino)!=self.created_signature:
            raise ValueError('Stage replaced')
        def budget():
            if self.clock()>=deadline:raise TimeoutError('Absolute arm deadline during output transport')
            if shutil.disk_usage(self.physical_root).free<=self.reserve_bytes:
                raise OSError('Physical output reserve reached')
        budget();started=self.clock()
        files=sorted(self.stage.iterdir(),key=lambda p:p.name)
        records={}
        for source in files:
            if not FILE.fullmatch(source.name) or source.name in ('.','..'):
                raise ValueError('Unscoped output filename')
            before=signature(source);target=self.destination/source.name
            if target.exists() or target.is_symlink():raise FileExistsError(target)
            source_hash=hashlib.sha256()
            # Exclusive final names are not acceptance until case.json completes.
            # Partial files and original stage are preserved if any check fails.
            with source.open('rb') as input_stream, target.open('xb') as output_stream:
                if (os.fstat(input_stream.fileno()).st_dev,os.fstat(input_stream.fileno()).st_ino)!=before[:2]:
                    raise ValueError('Source replaced before export')
                for chunk in iter(lambda:input_stream.read(1024*1024),b''):
                    budget();source_hash.update(chunk);output_stream.write(chunk)
                output_stream.flush();os.fsync(output_stream.fileno())
            if signature(source)!=before:raise ValueError('Source changed during export')
            verify_source=hashlib.sha256();verify_target=hashlib.sha256()
            for path,h in ((source,verify_source),(target,verify_target)):
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):budget();h.update(chunk)
            if signature(source)!=before or signature(target)[2]!=before[2]:raise ValueError('Output changed at final audit')
            expected=source_hash.hexdigest()
            if expected!=verify_source.hexdigest() or expected!=verify_target.hexdigest():
                raise ValueError('Output bytes differ at final audit')
            records[source.name]={'bytes':before[2],'sha256':expected}
        budget()
        if [p.name for p in sorted(self.stage.iterdir(),key=lambda p:p.name)]!=[p.name for p in files]:
            raise ValueError('Stage file set changed')
        self.exported=True
        return dict(status='verified_hash_exact_final_export_original_stage_preserved',stage=str(self.stage),destination=str(self.destination),
                    files=records,bytes=sum(v['bytes'] for v in records.values()),wall_s=self.clock()-started,
                    absolute_deadline_s=deadline,writer_exited=True,deleted_files=0)
