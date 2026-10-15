"""Fail-closed subprocess runner with sampled process-tree RSS telemetry.

RSS polling is a software guard, not an operating-system hard memory limit.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import ctypes
from pathlib import Path
from typing import Any


def _load_psutil():
    try:
        import psutil

        return psutil
    except ImportError:
        return None


def _tree(root: Any, psutil: Any) -> list[Any]:
    try:
        return [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return [root]


def _sample_tree(root: Any, psutil: Any) -> tuple[int, int, list[Any]]:
    processes = _tree(root, psutil)
    total = 0
    sampled = 0
    for process in processes:
        try:
            total += int(process.memory_info().rss)
            sampled += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total, sampled, processes


def _terminate_tree(root: Any, popen: subprocess.Popen, psutil: Any, grace_s: float = 0.5) -> bool:
    processes = _tree(root, psutil)
    for process in reversed(processes):
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if popen.poll() is not None and all(not _is_running(proc, psutil) for proc in processes[1:]):
            break
        time.sleep(0.02)
    for process in reversed(processes):
        if _is_running(process, psutil):
            try:
                process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
    try:
        popen.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        popen.kill()
        popen.wait()
    return popen.poll() is not None and all(not _is_running(proc, psutil) for proc in processes[1:])


def _is_running(process: Any, psutil: Any) -> bool:
    try:
        return bool(process.is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False


def _terminate_unmonitored_process_tree(popen: subprocess.Popen, grace_s: float = 0.5) -> bool:
    """Terminate the isolated process group/tree if psutil cannot create a handle."""
    if os.name == "nt":
        tree_pids = _windows_process_tree_pids(popen.pid)
        if tree_pids is None:
            try:
                popen.kill()
                popen.wait()
            except OSError:
                pass
            return False
        # Kill descendants first, then the root, so it cannot keep spawning.
        for pid in reversed(tree_pids):
            _windows_terminate_pid(pid)
        try:
            popen.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            popen.kill()
            popen.wait()
        deadline = time.monotonic() + max(1.0, grace_s)
        while time.monotonic() < deadline:
            if not any(_windows_pid_running(pid) for pid in tree_pids):
                return True
            time.sleep(0.03)
        return not any(_windows_pid_running(pid) for pid in tree_pids)

    try:
        os.killpg(popen.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    try:
        popen.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(popen.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        popen.wait()
    return popen.poll() is not None


def _windows_process_tree_pids(root_pid: int) -> list[int] | None:
    """Snapshot the root and current descendants using the Windows Toolhelp API."""
    if os.name != "nt":
        return None

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong), ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", ctypes.c_ulong), ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong), ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong), ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    snapshot_fn = kernel32.CreateToolhelp32Snapshot
    snapshot_fn.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    snapshot_fn.restype = ctypes.c_void_p
    first_fn = kernel32.Process32FirstW
    first_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32W)]
    first_fn.restype = ctypes.c_int
    next_fn = kernel32.Process32NextW
    next_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32W)]
    next_fn.restype = ctypes.c_int
    close_fn = kernel32.CloseHandle
    close_fn.argtypes = [ctypes.c_void_p]
    close_fn.restype = ctypes.c_int
    invalid_handle = ctypes.c_void_p(-1).value
    snapshot = snapshot_fn(0x00000002, 0)  # TH32CS_SNAPPROCESS
    if snapshot in (None, invalid_handle):
        return None
    children: dict[int, list[int]] = {}
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(ProcessEntry32W)
        ok = first_fn(snapshot, ctypes.byref(entry))
        while ok:
            children.setdefault(int(entry.th32ParentProcessID), []).append(int(entry.th32ProcessID))
            ok = next_fn(snapshot, ctypes.byref(entry))
    finally:
        close_fn(snapshot)
    result, frontier = [int(root_pid)], [int(root_pid)]
    seen = {int(root_pid)}
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, []):
            if child not in seen:
                seen.add(child)
                result.append(child)
                frontier.append(child)
    return result


def _windows_pid_running(pid: int) -> bool:
    if os.name != "nt":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    open_process.restype = ctypes.c_void_p
    wait = kernel32.WaitForSingleObject
    wait.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    wait.restype = ctypes.c_ulong
    close = kernel32.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    close.restype = ctypes.c_int
    handle = open_process(0x00100000 | 0x1000, 0, int(pid))  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        return wait(handle, 0) == 0x00000102  # WAIT_TIMEOUT means still active
    finally:
        close(handle)


def _windows_terminate_pid(pid: int) -> bool:
    if os.name != "nt":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    open_process.restype = ctypes.c_void_p
    terminate = kernel32.TerminateProcess
    terminate.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    terminate.restype = ctypes.c_int
    close = kernel32.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    close.restype = ctypes.c_int
    handle = open_process(0x0001 | 0x00100000, 0, int(pid))  # PROCESS_TERMINATE | SYNCHRONIZE
    if not handle:
        return not _windows_pid_running(pid)
    try:
        terminate(handle, 1)
        return True
    finally:
        close(handle)


def _base_result(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "return_code": None,
        "wall_seconds": 0.0,
        "peak_rss_bytes": None,
        "rss_sample_count": 0,
        "process_tree_terminated": False,
        "polling_limit_enforced": False,
        "os_hard_memory_limit_applied": False,
        "termination_reason": None,
    }


def run_monitored(
    command,
    *,
    cwd,
    env,
    timeout_s,
    rss_limit_bytes,
    sample_interval_s,
    output_dir,
) -> dict[str, Any]:
    """Run a child only when recursive RSS monitoring is available."""
    started = time.monotonic()
    psutil = _load_psutil()
    if psutil is None or not callable(getattr(psutil, "Process", None)):
        result = _base_result("MONITOR_UNAVAILABLE")
        result["wall_seconds"] = time.monotonic() - started
        result["termination_reason"] = "psutil_process_monitor_unavailable"
        return result

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = out / "stdout.log", out / "stderr.log"
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    kwargs = {"creationflags": creationflags} if os.name == "nt" else {"start_new_session": True}
    peak_rss = sample_count = 0
    termination_reason = None
    status = None
    terminated = False
    root = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, **kwargs)
        except OSError as exc:
            result = _base_result("LAUNCH_FAILED")
            result.update({"wall_seconds": time.monotonic() - started, "termination_reason": str(exc),
                           "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)})
            return result
        try:
            root = psutil.Process(proc.pid)
            while True:
                elapsed = time.monotonic() - started
                rss, sampled, _ = _sample_tree(root, psutil)
                if sampled:
                    peak_rss = max(peak_rss, rss)
                    sample_count += 1
                if rss_limit_bytes > 0 and sampled and rss > rss_limit_bytes:
                    status, termination_reason = "RESOURCE_LIMIT", "sampled_process_tree_rss_over_limit"
                    terminated = _terminate_tree(root, proc, psutil)
                    break
                if proc.poll() is not None:
                    break
                if sampled == 0:
                    status, termination_reason = "MONITOR_ERROR", "rss_sampling_unavailable"
                    terminated = (
                        _terminate_tree(root, proc, psutil)
                        if root is not None else _terminate_unmonitored_process_tree(proc)
                    )
                    break
                if timeout_s > 0 and elapsed >= timeout_s:
                    status, termination_reason = "TIMEOUT", "wall_clock_timeout"
                    terminated = _terminate_tree(root, proc, psutil)
                    break
                time.sleep(max(0.01, float(sample_interval_s)))
        except Exception as exc:
            status, termination_reason = "MONITOR_ERROR", f"{type(exc).__name__}: {exc}"
            if root is not None:
                terminated = _terminate_tree(root, proc, psutil)
            else:
                terminated = _terminate_unmonitored_process_tree(proc)
        return_code = proc.wait()

    if status is None:
        status = "COMPLETED" if return_code == 0 else "METHOD_FAILED"
    result = _base_result(status)
    result.update({
        "return_code": return_code,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss if sample_count else None,
        "rss_sample_count": sample_count,
        "process_tree_terminated": terminated,
        "polling_limit_enforced": bool(sample_count),
        "os_hard_memory_limit_applied": False,
        "termination_reason": termination_reason,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": list(command),
        "rss_limit_bytes": int(rss_limit_bytes),
        "sample_interval_s": float(sample_interval_s),
    })
    out.joinpath("resource_status.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
