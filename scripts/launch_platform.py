"""Launch the verified checkout on loopback without adopting unknown listeners."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser


def checkout_identity(root):
    root=Path(root)
    files=[root/'app.py',*sorted((root/'src/cfquant').glob('*.py'))]
    value=hashlib.sha256()
    for path in files:
        value.update(path.relative_to(root).as_posix().encode());value.update(path.read_bytes())
    return value.hexdigest()


def healthy(port):
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/_stcore/health',timeout=1) as response:
            return response.status==200 and response.read(32).strip()==b'ok'
    except (OSError,ValueError):return False


def available_port(preferred):
    for port in range(preferred,min(preferred+100,65536)):
        with socket.socket() as sock:
            if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            try:sock.bind(('127.0.0.1',port));return port
            except OSError:continue
    raise RuntimeError('No available loopback port in requested range')


def _windows_native_listener_owns(port,pid):
    """Read the native owner table without requiring CIM/WMI permissions."""
    try:
        import ctypes
        from ctypes import wintypes
        class TcpRow(ctypes.Structure):
            _fields_=[(name,wintypes.DWORD) for name in
                ('state','local_addr','local_port','remote_addr','remote_port','pid')]
        query=ctypes.WinDLL('iphlpapi',use_last_error=True).GetExtendedTcpTable
        query.argtypes=[ctypes.c_void_p,ctypes.POINTER(wintypes.DWORD),
            wintypes.BOOL,wintypes.ULONG,wintypes.ULONG,wintypes.ULONG]
        query.restype=wintypes.DWORD
        size=wintypes.DWORD(0)
        # TCP_TABLE_OWNER_PID_LISTENER is 3; ERROR_INSUFFICIENT_BUFFER is 122.
        status=query(None,ctypes.byref(size),False,socket.AF_INET,3,0)
        if status not in (0,122):return None
        for _ in range(2):
            if not 4<=size.value<=1024*1024:return None
            buffer=ctypes.create_string_buffer(size.value)
            status=query(buffer,ctypes.byref(size),False,socket.AF_INET,3,0)
            if status==122:continue
            if status!=0 or not 4<=size.value<=len(buffer):return None
            count=wintypes.DWORD.from_buffer(buffer).value
            stride=ctypes.sizeof(TcpRow)
            if count>(size.value-4)//stride:return None
            owners=[]
            for index in range(count):
                row=TcpRow.from_buffer(buffer,4+index*stride)
                if (row.state==2 and row.local_addr==0x0100007F and
                    socket.ntohs(row.local_port & 0xffff)==port):
                    owners.append(int(row.pid))
            return bool(owners) and all(owner==pid for owner in owners)
        return None
    except (AttributeError,OSError,ValueError,OverflowError):return None


def _windows_listener_owns(port,pid):
    """Use a native table, with a bounded hidden OS query as fallback."""
    native=_windows_native_listener_owns(port,pid)
    if native is not None:return native
    command=("$ErrorActionPreference='Stop'; "
        f"$listeners=@(Get-NetTCPConnection -State Listen -LocalPort {port} -ErrorAction Stop | "
        "Where-Object {$_.LocalAddress -eq '127.0.0.1'}); "
        f"if($listeners.Count -gt 0 -and @($listeners | Where-Object {{$_.OwningProcess -ne {pid}}}).Count -eq 0)"
        "{'MATCH'}else{'MISMATCH'}")
    try:
        result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
            capture_output=True,text=True,timeout=3,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return result.returncode==0 and result.stdout.strip()=='MATCH'
    except (OSError,subprocess.TimeoutExpired,UnicodeError):return False


def _linux_listener_owns(port,pid):
    """Prove a fresh child's IPv4 loopback socket ownership using procfs."""
    try:
        inodes=set()
        for row in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            fields=row.split()
            address,encoded_port=fields[1].split(':')
            if address=='0100007F' and int(encoded_port,16)==port and fields[3]=='0A':
                inodes.add(fields[9])
        if not inodes:return False
        child_sockets=set()
        for fd in (Path('/proc')/str(pid)/'fd').iterdir():
            try:
                link=os.readlink(fd)
                if link.startswith('socket:[') and link.endswith(']'):
                    child_sockets.add(link[8:-1])
            except OSError:continue
        return inodes.issubset(child_sockets)
    except (OSError,ValueError,IndexError):return False


def listener_belongs_to(port,pid):
    if type(port) is not int or not 1024<=port<=65535 or type(pid) is not int or pid<=0:
        return False
    if os.name=='nt':return _windows_listener_owns(port,pid)
    if sys.platform.startswith('linux'):return _linux_listener_owns(port,pid)
    return False


def reusable_receipt(root, identity):
    path=Path(root)/'runs/platform_server.json'
    try:
        receipt=json.loads(path.read_text(encoding='utf-8'))
        if receipt.get('identity')!=identity or receipt.get('checkout')!=str(Path(root).resolve()):return None
        port=receipt['port']
        pid=receipt.get('pid')
        if type(port) is not int or not 1024<=port<=65535 or type(pid) is not int or pid<=0:return None
        # Outside Windows, start a fresh port instead of trusting old receipts.
        if os.name!='nt':return None
        return receipt if healthy(port) and listener_belongs_to(port,pid) else None
    except (OSError,ValueError,KeyError):return None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8501)
    parser.add_argument('--no-browser',action='store_true')
    args,extra=parser.parse_known_args(argv)
    if not 1024<=args.port<=65435:parser.error('port must be between 1024 and 65435')
    if any(x.startswith(('--server.address','--server.port')) for x in extra):
        parser.error('Use --port; server binding stays on loopback')
    root=Path(__file__).resolve().parents[1]
    identity=checkout_identity(root)
    version=re.search(r'__version__\s*=\s*[\"\']([^\"\']+)',(root/'src/cfquant/__init__.py').read_text()).group(1)
    print(f'Qingxu v{version}\nCheckout: {root}',flush=True)
    receipt=reusable_receipt(root,identity)
    if receipt:
        url=f"http://127.0.0.1:{receipt['port']}"
        print('Verified checkout receipt + listener PID + healthy service: '+url,flush=True)
        if not args.no_browser:webbrowser.open(url)
        return 0
    logs=root/'runs/launcher_logs';logs.mkdir(parents=True,exist_ok=True)
    for attempt in range(5):
        port=available_port(args.port+attempt)
        path=logs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.log')
        env=os.environ.copy();env['PYTHONPATH']=str(root/'src')
        command=[sys.executable,'-m','streamlit','run',str(root/'app.py'),
                 '--server.address','127.0.0.1','--server.port',str(port),
                 '--server.headless','true','--browser.gatherUsageStats','false',*extra]
        with path.open('wb') as stream:
            child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
                 stdout=stream,stderr=subprocess.STDOUT,
                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        for _ in range(120):
            # A healthy port alone may belong to a process racing our bind.
            if child.poll() is not None:break
            if healthy(port) and listener_belongs_to(port,child.pid) and child.poll() is None:
                receipt={'checkout':str(root.resolve()),'version':version,'identity':identity,
                         'port':port,'pid':child.pid,'started_at':datetime.now(timezone.utc).isoformat(),
                         'log':str(path)}
                (root/'runs/platform_server.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
                url=f'http://127.0.0.1:{port}'
                print('Ready: '+url+'\nLog: '+str(path),flush=True)
                if not args.no_browser:webbrowser.open(url)
                return 0
            time.sleep(.25)
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill();child.wait(timeout=5)
        message=path.read_text(encoding='utf-8',errors='replace')
        if 'already in use' not in message.lower():
            print('Launch failed; inspect '+str(path),file=sys.stderr)
            return child.returncode or 1
    print('Port raced repeatedly; retry launch.',file=sys.stderr)
    return 1


if __name__=='__main__':raise SystemExit(main())
