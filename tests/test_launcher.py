import importlib.util
import json
from pathlib import Path
import socket
import subprocess
from unittest.mock import Mock

import pytest

spec=importlib.util.spec_from_file_location('qingxu_launcher',Path(__file__).resolve().parents[1]/'scripts/launch_platform.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


def test_unknown_listener_is_not_adopted_and_source_change_invalidates_receipt(tmp_path,monkeypatch):
    (tmp_path/'src/cfquant').mkdir(parents=True);(tmp_path/'runs').mkdir()
    (tmp_path/'app.py').write_text('first');(tmp_path/'src/cfquant/a.py').write_text('a')
    identity=launcher.checkout_identity(tmp_path)
    monkeypatch.setattr(launcher,'healthy',lambda port:True)
    monkeypatch.setattr(launcher,'listener_belongs_to',lambda port,pid:pid==1234)
    assert launcher.reusable_receipt(tmp_path,identity) is None
    (tmp_path/'runs/platform_server.json').write_text(json.dumps({'identity':identity,'checkout':str(tmp_path.resolve()),'port':8501,'pid':1234}))
    if launcher.os.name=='nt':assert launcher.reusable_receipt(tmp_path,identity)['port']==8501
    else:assert launcher.reusable_receipt(tmp_path,identity) is None
    (tmp_path/'app.py').write_text('new')
    assert launcher.reusable_receipt(tmp_path,launcher.checkout_identity(tmp_path)) is None


def test_occupied_port_is_preserved():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
        listener.listen()
        assert launcher.available_port(port)!=port
        assert listener.getsockname()[1]==port


def test_stale_receipt_cannot_adopt_healthy_port_of_another_process(tmp_path,monkeypatch):
    (tmp_path/'runs').mkdir()
    identity='same-checkout'
    receipt={'identity':identity,'checkout':str(tmp_path.resolve()),'port':8501,'pid':1234}
    (tmp_path/'runs/platform_server.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(launcher,'healthy',lambda port:True)
    monkeypatch.setattr(launcher,'listener_belongs_to',lambda port,pid:False)
    assert launcher.reusable_receipt(tmp_path,identity) is None
    receipt.pop('pid')
    (tmp_path/'runs/platform_server.json').write_text(json.dumps(receipt))
    assert launcher.reusable_receipt(tmp_path,identity) is None


def test_hidden_windows_os_query_is_bounded_and_requires_exact_owner_result(monkeypatch):
    monkeypatch.setattr(launcher,'_windows_native_listener_owns',lambda port,pid:None)
    run=Mock(return_value=Mock(returncode=0,stdout='MATCH\n'))
    monkeypatch.setattr(launcher.subprocess,'run',run)
    assert launcher._windows_listener_owns(8501,1234)
    args,kwargs=run.call_args
    assert args[0][:3]==['powershell.exe','-NoProfile','-NonInteractive']
    assert 'Get-NetTCPConnection -State Listen -LocalPort 8501' in args[0][-1]
    assert 'OwningProcess -ne 1234' in args[0][-1]
    assert kwargs['timeout']==3 and kwargs['creationflags']==getattr(subprocess,'CREATE_NO_WINDOW',0)
    run.return_value=Mock(returncode=0,stdout='MISMATCH\n')
    assert not launcher._windows_listener_owns(8501,1234)
    run.side_effect=subprocess.TimeoutExpired('hidden query',3)
    assert not launcher._windows_listener_owns(8501,1234)


def test_native_owner_mismatch_is_not_overridden_by_fallback(monkeypatch):
    monkeypatch.setattr(launcher,'_windows_native_listener_owns',lambda port,pid:False)
    run=Mock(return_value=Mock(returncode=0,stdout='MATCH\n'))
    monkeypatch.setattr(launcher.subprocess,'run',run)
    assert not launcher._windows_listener_owns(8501,1234)
    run.assert_not_called()


@pytest.mark.skipif(launcher.os.name!='nt',reason='Windows native owner table')
def test_native_windows_owner_table_verifies_real_loopback_listener():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));listener.listen()
        port=listener.getsockname()[1]
        assert launcher._windows_native_listener_owns(port,launcher.os.getpid()) is True
        assert launcher._windows_native_listener_owns(port,launcher.os.getpid()+1) is False
        assert launcher.listener_belongs_to(port,launcher.os.getpid())


@pytest.mark.parametrize('port,pid',[(True,12),(8501,True),(8501,0),('8501',1234),(8501,'1234')])
def test_owner_query_rejects_noninteger_inputs_before_os_call(port,pid,monkeypatch):
    query=Mock()
    monkeypatch.setattr(launcher,'_windows_listener_owns',query)
    assert not launcher.listener_belongs_to(port,pid)
    query.assert_not_called()


def fake_checkout(root,monkeypatch):
    (root/'src/cfquant').mkdir(parents=True)
    (root/'app.py').write_text('app')
    (root/'src/cfquant/__init__.py').write_text('__version__ = "2.3.0"\n')
    monkeypatch.setattr(launcher,'__file__',str(root/'scripts/launch_platform.py'))
    monkeypatch.setattr(launcher,'healthy',lambda port:True)
    monkeypatch.setattr(launcher.time,'sleep',lambda duration:None)


def test_fresh_launch_waits_for_child_listener_ownership_before_ready(tmp_path,monkeypatch):
    fake_checkout(tmp_path,monkeypatch)
    child=Mock(pid=4321)
    child.poll.return_value=None
    popen=Mock(return_value=child)
    owner=Mock(side_effect=[False,True])
    browser=Mock()
    monkeypatch.setattr(launcher,'available_port',lambda preferred:8502)
    monkeypatch.setattr(launcher.subprocess,'Popen',popen)
    monkeypatch.setattr(launcher,'listener_belongs_to',owner)
    monkeypatch.setattr(launcher.webbrowser,'open',browser)
    assert launcher.main(['--port','8501'])==0
    assert owner.call_count==2
    browser.assert_called_once_with('http://127.0.0.1:8502')
    receipt=json.loads((tmp_path/'runs/platform_server.json').read_text())
    assert receipt['pid']==4321 and receipt['port']==8502
    assert '127.0.0.1' in popen.call_args.args[0]


def test_racing_unknown_listener_is_not_ready_and_next_port_retries(tmp_path,monkeypatch):
    fake_checkout(tmp_path,monkeypatch)
    first=Mock(pid=1001)
    first.poll.side_effect=[None,7,7]
    first.returncode=7
    second=Mock(pid=1002)
    second.poll.return_value=None
    children=[first,second]
    def popen(command,**kwargs):
        kwargs['stdout'].write(b'Port is already in use\n')
        return children.pop(0)
    owner=Mock(side_effect=lambda port,pid:pid==1002)
    browser=Mock()
    monkeypatch.setattr(launcher,'available_port',lambda preferred:preferred)
    monkeypatch.setattr(launcher.subprocess,'Popen',popen)
    monkeypatch.setattr(launcher,'listener_belongs_to',owner)
    monkeypatch.setattr(launcher.webbrowser,'open',browser)
    assert launcher.main(['--port','8501'])==0
    browser.assert_called_once_with('http://127.0.0.1:8502')
    receipt=json.loads((tmp_path/'runs/platform_server.json').read_text())
    assert receipt['pid']==1002
