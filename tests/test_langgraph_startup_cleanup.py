"""Startup rollback uses real, isolated subprocesses without a model or server."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from EvoScientist.langgraph_dev import manager


@pytest.fixture
def startup_env(monkeypatch, tmp_path, runtime_paths):
    """Replace external setup, but retain real Popen, shutdown and runtime files."""
    real_popen = subprocess.Popen
    processes = []
    by_port = {}
    state = SimpleNamespace(health=lambda port, proc: False)
    config = tmp_path / "langgraph.json"
    config.write_text("{}")
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)
    monkeypatch.setattr(manager, "_PROCESS_DEPLOY_MODE", None)
    monkeypatch.setattr(manager, "_STARTUP_STATE", None, raising=False)
    monkeypatch.setattr(manager, "_START_IN_PROGRESS", False, raising=False)
    monkeypatch.setattr(manager, "_langgraph_exe", lambda: sys.executable)
    monkeypatch.setattr(manager, "_packaged_langgraph_config", lambda: config)
    monkeypatch.setattr(manager, "_is_port_occupied", lambda *a, **kw: False)
    monkeypatch.setattr(manager, "_wait_for_port_bindable", lambda *a, **kw: True)
    monkeypatch.setattr("EvoScientist.mcp.client.load_mcp_config", lambda: {})
    monkeypatch.setattr("EvoScientist.setup.research_env.agent_python", lambda: None)

    def spawn(args, **kwargs):
        proc = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
        )
        processes.append(proc)
        by_port[int(args[args.index("--port") + 1])] = proc
        return proc

    def health(*, port, **kwargs):
        if port not in by_port:
            return False
        return state.health(port, by_port[port])

    monkeypatch.setattr(manager.subprocess, "Popen", spawn)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", health)

    def previous(*, tracked):
        proc = real_popen([sys.executable, "-c", "import time; time.sleep(60)"])
        processes.append(proc)
        runtime_paths.pid_dir.mkdir(parents=True, exist_ok=True)
        runtime_paths.pid_file.write_text(str(proc.pid))
        manager._write_workspace_sidecar(tmp_path / "previous", proc.pid)
        if tracked:
            manager._PROCESS = proc
            manager._PROCESS_WORKSPACE = tmp_path / "previous"
            manager._PROCESS_DEPLOY_MODE = True
        return proc, {
            path: path.read_bytes()
            for path in (runtime_paths.pid_file, runtime_paths.workspace_sidecar)
        }

    state.start = lambda port=16174: manager.start_langgraph_dev(tmp_path, port=port)
    state.processes = processes
    state.real_popen = real_popen
    state.by_port = by_port
    state.runtime = runtime_paths
    state.previous = previous
    yield state
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_interrupted_health_wait_stops_real_child(startup_env, error_type):
    """Every exceptional exit after Popen must reap the call's own child."""
    error = error_type("interrupted")

    def interrupt(port, proc):
        raise error

    startup_env.health = interrupt
    with pytest.raises(error_type) as caught:
        startup_env.start()
    assert caught.value is error
    assert startup_env.processes[0].poll() is not None
    assert manager._PROCESS is None
    assert not startup_env.runtime.pid_file.exists()
    assert not startup_env.runtime.workspace_sidecar.exists()


def test_interrupted_bookkeeping_stops_real_child(startup_env, monkeypatch):
    """The cleanup guard starts before sidecar work, not only around polling."""
    error = KeyboardInterrupt("sidecar interrupted")

    def interrupt(**kwargs):
        raise error

    monkeypatch.setattr(manager, "_write_workspace_sidecar", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        startup_env.start()
    assert caught.value is error
    assert startup_env.processes[0].poll() is not None
    assert manager._PROCESS is None
    assert not startup_env.runtime.pid_file.exists()


@pytest.mark.parametrize("tracked", [False, True])
def test_interruption_preserves_previous_live_server(startup_env, tracked):
    """Keepalive records matter even when the previous server is untracked."""
    previous, records = startup_env.previous(tracked=tracked)

    def interrupt(port, proc):
        raise KeyboardInterrupt

    startup_env.health = interrupt
    with pytest.raises(KeyboardInterrupt):
        startup_env.start()
    assert previous.poll() is None
    assert startup_env.processes[-1].poll() is not None
    assert manager._PROCESS is (previous if tracked else None)
    assert {path: path.read_bytes() for path in records} == records


def test_waiting_start_succeeds_after_older_interruption(startup_env):
    """Another thread starts only after the first call has finished rollback."""
    first_polling = threading.Event()
    second_polling = threading.Event()
    release = threading.Event()

    def health(port, proc):
        if port == 16174:
            first_polling.set()
            assert release.wait(5)
            raise KeyboardInterrupt
        second_polling.set()
        return True

    startup_env.health = health
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(startup_env.start)
        assert first_polling.wait(5)
        second = pool.submit(startup_env.start, 16175)
        try:
            assert not second_polling.wait(0.2)
        finally:
            release.set()
        with pytest.raises(KeyboardInterrupt):
            first.result(timeout=5)
        new = second.result(timeout=5)
    old = startup_env.by_port[16174]
    assert old.poll() is not None
    assert new.poll() is None
    assert manager._PROCESS is new
    assert startup_env.runtime.pid_file.read_text() == str(new.pid)
    assert (
        json.loads(startup_env.runtime.workspace_sidecar.read_text())["pid"] == new.pid
    )


def test_explicit_old_stop_keeps_newer_runtime_files(startup_env):
    """The shared stop helper must not unlink another process's records."""
    startup_env.health = lambda port, proc: True
    old = startup_env.start()
    new = startup_env.start(16175)
    manager.stop_langgraph_dev(old)
    assert old.poll() is not None
    assert new.poll() is None
    assert manager._PROCESS is new
    assert startup_env.runtime.pid_file.read_text() == str(new.pid)
    assert (
        json.loads(startup_env.runtime.workspace_sidecar.read_text())["pid"] == new.pid
    )


def test_stop_during_health_wait_stays_responsive(startup_env):
    """A second thread can cancel startup without waiting out the health budget."""
    previous, records = startup_env.previous(tracked=False)
    polling = threading.Event()
    stopped = threading.Event()

    def health(port, proc):
        polling.set()
        assert stopped.wait(5), "stop was blocked by the startup health wait"
        raise KeyboardInterrupt

    startup_env.health = health
    with ThreadPoolExecutor(max_workers=2) as pool:
        starting = pool.submit(startup_env.start)
        assert polling.wait(5)
        stopping = pool.submit(manager.stop_inflight_owned_server)
        try:
            assert stopping.result(timeout=5) == startup_env.by_port[16174].pid
        finally:
            stopped.set()
        with pytest.raises(KeyboardInterrupt):
            starting.result(timeout=5)
    assert previous.poll() is None
    assert startup_env.by_port[16174].poll() is not None
    assert {path: path.read_bytes() for path in records} == records


def test_two_failed_serialized_starts_restore_live_predecessor(startup_env):
    """Queued failed starts must each preserve the live keepalive predecessor."""
    previous, records = startup_env.previous(tracked=False)
    first_polling = threading.Event()
    second_polling = threading.Event()
    release = threading.Event()

    def health(port, proc):
        if port == 16174:
            first_polling.set()
            assert release.wait(5)
        else:
            second_polling.set()
        raise KeyboardInterrupt

    startup_env.health = health
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(startup_env.start)
        assert first_polling.wait(5)
        second = pool.submit(startup_env.start, 16175)
        try:
            assert not second_polling.wait(0.2)
        finally:
            release.set()
        with pytest.raises(KeyboardInterrupt):
            first.result(timeout=5)
        with pytest.raises(KeyboardInterrupt):
            second.result(timeout=5)
    assert previous.poll() is None
    assert all(proc.poll() is not None for proc in startup_env.processes[1:])
    assert manager._PROCESS is None
    assert {path: path.read_bytes() for path in records} == records


def test_reentrant_start_rejected_without_disturbing_active_start(startup_env):
    """A recursive caller cannot replace the one manager-owned startup."""
    previous, records = startup_env.previous(tracked=False)

    def health(port, proc):
        if port == 16175:
            return True
        with pytest.raises(RuntimeError, match="already in progress"):
            startup_env.start(16175)
        assert proc.poll() is None
        assert manager._PROCESS is proc
        assert startup_env.runtime.pid_file.read_text() == str(proc.pid)
        raise KeyboardInterrupt

    startup_env.health = health
    with pytest.raises(KeyboardInterrupt):
        startup_env.start()
    assert len(startup_env.processes) == 2  # previous plus this call's child
    assert previous.poll() is None
    assert {path: path.read_bytes() for path in records} == records


def test_ensure_can_reenter_lifecycle_lock(startup_env, monkeypatch, tmp_path):
    """ensure's outer lock and the direct-start lock share one reentrant owner."""
    from EvoScientist.config import EvoScientistConfig

    config = EvoScientistConfig(enable_async_subagents=True, langgraph_dev_port=16174)
    monkeypatch.setattr(manager.atexit, "register", lambda *a, **kw: None)
    startup_env.health = lambda port, proc: True
    proc = manager.ensure_langgraph_dev(config, workspace_dir=tmp_path)
    assert proc is startup_env.by_port[16174]
    assert proc.poll() is None


def test_inflight_stop_ignores_prior_healthy_process_before_spawn(startup_env):
    """Cancellation must not stop an older server while a new start is waiting."""
    previous, records = startup_env.previous(tracked=True)
    assert manager.stop_inflight_owned_server() is None
    assert previous.poll() is None
    assert manager._PROCESS is previous
    assert {path: path.read_bytes() for path in records} == records


def test_success_keeps_process_and_records(startup_env):
    """Successful startup still transfers a live process handle to its caller."""
    startup_env.health = lambda port, proc: True
    proc = startup_env.start()
    assert proc.poll() is None
    assert manager._PROCESS is proc
    assert manager._STARTUP_STATE is None
    assert startup_env.runtime.pid_file.read_text() == str(proc.pid)
    assert (
        json.loads(startup_env.runtime.workspace_sidecar.read_text())["pid"] == proc.pid
    )


def test_interrupt_while_waiting_for_lifecycle_lock_keeps_prior_state(startup_env):
    """Ctrl+C while another interpreter owns startup must not spawn or stop anything."""
    import _thread

    previous, records = startup_env.previous(tracked=True)
    code = (
        "import sys; from filelock import FileLock; "
        "lock=FileLock(sys.argv[1]); lock.acquire(); "
        "print('locked', flush=True); sys.stdin.read(1); lock.release()"
    )
    other = startup_env.real_popen(
        [sys.executable, "-c", code, str(startup_env.runtime.lock_file)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    timer = threading.Timer(0.1, _thread.interrupt_main)
    try:
        assert other.stdout.readline().strip() == "locked"
        timer.start()
        with pytest.raises(KeyboardInterrupt):
            startup_env.start()
        assert startup_env.processes == [previous]
        assert previous.poll() is None
        assert manager._PROCESS is previous
        assert not manager._START_IN_PROGRESS
        assert {path: path.read_bytes() for path in records} == records
    finally:
        timer.cancel()
        timer.join()
        other.communicate("x", timeout=5)


def test_interrupted_startup_lock_check_does_not_latch(startup_env, monkeypatch):
    """An interrupt on leaving the entry check must not leave startup latched."""
    lock = manager._LOCK

    class InterruptOnExit:
        def __enter__(self):
            lock.acquire()

        def __exit__(self, *exc):
            lock.release()
            raise KeyboardInterrupt

    monkeypatch.setattr(manager, "_LOCK", InterruptOnExit())
    with pytest.raises(KeyboardInterrupt), manager._startup_lock():
        pytest.fail("entry was interrupted")
    assert not manager._START_IN_PROGRESS


def test_cross_process_start_cannot_be_overwritten_by_rollback(
    startup_env, monkeypatch, tmp_path
):
    """Another interpreter can publish between existence check and restoration."""
    previous, _ = startup_env.previous(tracked=False)
    ready = tmp_path / "other_ready"
    stop = tmp_path / "other_stop"
    code = r"""
import json, pathlib, subprocess, sys, time
from unittest.mock import patch
from EvoScientist.langgraph_dev import manager
root = pathlib.Path(sys.argv[1])
manager.RUNTIME = manager.LanggraphRuntimePaths.for_directory(root / "runtime")
real_popen = subprocess.Popen
child = None

def spawn(*args, **kwargs):
    global child
    child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
    return child
try:
    with (
        patch.object(manager, "_langgraph_exe", return_value=sys.executable),
        patch.object(manager, "_packaged_langgraph_config", return_value=root / "langgraph.json"),
        patch.object(manager, "_is_port_occupied", return_value=False),
        patch.object(manager, "_wait_for_port_bindable", return_value=True),
        patch.object(manager, "is_langgraph_dev_running", return_value=True),
        patch.object(manager.subprocess, "Popen", side_effect=spawn),
        patch("EvoScientist.mcp.client.load_mcp_config", return_value={}),
        patch("EvoScientist.setup.research_env.agent_python", return_value=None),
    ):
        (root / "other_attempting").touch()
        proc = manager.start_langgraph_dev(root, port=16175)
        (root / "other_ready").write_text(str(proc.pid))
        deadline = time.monotonic() + 15
        while not (root / "other_stop").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
finally:
    if child is not None:
        manager.stop_langgraph_dev(child)
"""
    other = None
    blocked = []
    replace = manager._replace_runtime_record

    def interleave(path, content):
        nonlocal other
        if (
            path == startup_env.runtime.pid_file
            and content == str(previous.pid).encode()
        ):
            # The failing startup has just observed that its PID path is absent.
            # Publish a successful new startup from a truly separate interpreter.
            other = startup_env.real_popen([sys.executable, "-c", code, str(tmp_path)])
            import time

            deadline = time.monotonic() + 10
            while (
                not (tmp_path / "other_attempting").exists()
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert (tmp_path / "other_attempting").exists(), (
                "other startup never attempted"
            )
            time.sleep(0.2)
            blocked.append(not ready.exists())
        return replace(path, content)

    monkeypatch.setattr(manager, "_replace_runtime_record", interleave)
    startup_env.health = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    try:
        with pytest.raises(KeyboardInterrupt):
            startup_env.start()
        import time

        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "other startup did not resume"
        assert blocked == [True], "other start published during active rollback"
        newer_pid = int(ready.read_text())
        recorded_pid = int(startup_env.runtime.pid_file.read_text())
        sidecar_pid = json.loads(startup_env.runtime.workspace_sidecar.read_text())[
            "pid"
        ]
        assert recorded_pid == sidecar_pid == newer_pid, (
            f"rollback overwrote newer live startup: pid_file={recorded_pid}, "
            f"sidecar={sidecar_pid}, newer={newer_pid}, previous={previous.pid}"
        )
    finally:
        stop.touch()
        if other is not None:
            other.wait(timeout=10)


@pytest.mark.skipif(
    sys.platform != "linux", reason="uses Linux subreaper to reap test orphans"
)
def test_startup_rollback_kills_ignoring_descendant(startup_env, monkeypatch, tmp_path):
    """Fast parent exit must not exempt a stubborn child from kill escalation."""
    import ctypes
    import os
    import time

    import psutil

    libc = None
    old_subreaper = ctypes.c_int()
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        assert libc.prctl(37, ctypes.byref(old_subreaper), 0, 0, 0) == 0
        assert libc.prctl(36, 1, 0, 0, 0) == 0

    ready = tmp_path / "descendant_ready"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); time.sleep(60)"
    )

    def spawn(args, **kwargs):
        proc = startup_env.real_popen([sys.executable, "-c", parent_code], **kwargs)
        startup_env.processes.append(proc)
        startup_env.by_port[int(args[args.index("--port") + 1])] = proc
        return proc

    def health(*args):
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "descendant did not initialize"
        raise KeyboardInterrupt

    monkeypatch.setattr(manager.subprocess, "Popen", spawn)
    startup_env.health = health
    try:
        with pytest.raises(KeyboardInterrupt):
            startup_env.start()
        pid = int(ready.read_text())
        child = psutil.Process(pid)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE, (
            f"SIGTERM-ignoring descendant {pid} remains alive after startup rollback"
        )
    finally:
        if ready.exists():
            pid = int(ready.read_text())
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass
            if libc is not None:
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
        if libc is not None:
            assert libc.prctl(36, old_subreaper.value, 0, 0, 0) == 0


def test_restart_after_external_stop_keeps_live_predecessor(startup_env):
    """A stopped in-flight start still carries the keepalive rollback chain."""
    previous, records = startup_env.previous(tracked=False)
    first_polling = threading.Event()
    finish_first = threading.Event()

    def health(port, proc):
        if port == 16174:
            first_polling.set()
            assert finish_first.wait(5)
        raise KeyboardInterrupt

    startup_env.health = health
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(startup_env.start)
        assert first_polling.wait(5)
        manager.stop_inflight_owned_server()
        try:
            assert {path: path.read_bytes() for path in records} == records
            second = pool.submit(startup_env.start, 16175)
        finally:
            finish_first.set()
        with pytest.raises(KeyboardInterrupt):
            first.result(timeout=5)
        with pytest.raises(KeyboardInterrupt):
            second.result(timeout=5)
    assert previous.poll() is None
    assert {path: path.read_bytes() for path in records} == records


def test_two_failed_cross_process_starts_keep_live_predecessor(startup_env, tmp_path):
    """The recovery lineage must not disappear at an interpreter boundary."""
    import time

    previous, records = startup_env.previous(tracked=False)
    code = r"""
import pathlib, subprocess, sys, time
from unittest.mock import patch
from EvoScientist.langgraph_dev import manager
root = pathlib.Path(sys.argv[1])
name = sys.argv[2]
manager.RUNTIME = manager.LanggraphRuntimePaths.for_directory(root / "runtime")
real_popen = subprocess.Popen
children = []
def spawn(*args, **kwargs):
    child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
    children.append(child)
    return child
checks = 0
def health(**kwargs):
    global checks
    checks += 1
    if checks == 1:
        return False
    (root / (name + "_ready")).write_text(str(children[0].pid))
    deadline = time.monotonic() + 15
    while not (root / (name + "_fail")).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    raise KeyboardInterrupt
try:
    with (
        patch.object(manager, "_langgraph_exe", return_value=sys.executable),
        patch.object(manager, "_packaged_langgraph_config", return_value=root / "langgraph.json"),
        patch.object(manager, "_is_port_occupied", return_value=False),
        patch.object(manager, "_wait_for_port_bindable", return_value=True),
        patch.object(manager, "is_langgraph_dev_running", side_effect=health),
        patch.object(manager.subprocess, "Popen", side_effect=spawn),
        patch("EvoScientist.mcp.client.load_mcp_config", return_value={}),
        patch("EvoScientist.setup.research_env.agent_python", return_value=None),
    ):
        (root / (name + "_attempting")).touch()
        manager.start_langgraph_dev(root, port=int(sys.argv[3]))
except KeyboardInterrupt:
    pass
finally:
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
"""
    procs = []

    def start(name, port, wait_ready=True):
        proc = startup_env.real_popen(
            [sys.executable, "-c", code, str(tmp_path), name, str(port)]
        )
        procs.append(proc)
        deadline = time.monotonic() + 10
        marker = tmp_path / (name + ("_ready" if wait_ready else "_attempting"))
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        return proc

    try:
        a = start("a", 16174)
        b = start("b", 16175, wait_ready=False)
        time.sleep(0.2)
        assert not (tmp_path / "b_ready").exists(), "second start overlapped the first"
        (tmp_path / "a_fail").touch()
        a.wait(timeout=10)
        deadline = time.monotonic() + 10
        while not (tmp_path / "b_ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "b_ready").exists(), "second start never resumed"
        (tmp_path / "b_fail").touch()
        b.wait(timeout=10)
        assert previous.poll() is None
        assert {path: path.read_bytes() for path in records} == records
    finally:
        for name in ("a", "b"):
            (tmp_path / (name + "_fail")).touch()
        for proc in procs:
            proc.wait(timeout=10)


def test_interrupt_at_success_handoff_restores_predecessor(startup_env):
    """Ctrl+C before the Popen reaches its caller still rolls back this call."""
    import inspect

    previous, records = startup_env.previous(tracked=False)
    startup_env.health = lambda *args: True
    fn = inspect.unwrap(manager.start_langgraph_dev)
    lines, first = inspect.getsourcelines(fn)
    target = next(first + i for i, s in enumerate(lines) if s.strip() == "return proc")

    def interrupt(frame, event, arg):
        if frame.f_code is fn.__code__ and event == "line" and frame.f_lineno == target:
            raise KeyboardInterrupt("success handoff interrupted")
        return interrupt

    previous_trace = sys.gettrace()
    sys.settrace(interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            startup_env.start()
    finally:
        sys.settrace(previous_trace)
    assert previous.poll() is None
    assert startup_env.processes[-1].poll() is not None
    assert {path: path.read_bytes() for path in records} == records
