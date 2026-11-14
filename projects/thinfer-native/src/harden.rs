//! Make the running server's memory hard (not impossible) to read from
//! outside: job plaintext lives in it while a job runs. An administrator /
//! root can still get in; an ordinary process of the same user cannot.
//!
//! - Windows: the process object's DACL grants its owner only synchronize,
//!   query and terminate (still listable and stoppable with Task Manager,
//!   `taskkill`, `Stop-Process`), so another process of the same user cannot
//!   open it to read or
//!   write its memory, inject a thread or duplicate its handles. The process
//!   itself keeps full access through its pseudo handle.
//! - Linux: not dumpable, which blocks same-user ptrace and `/proc/<pid>/mem`
//!   reads and disables core dumps.
//! - macOS: debugger attach denied.

/// Apply the platform's hardening to the current process. Best effort: a
/// failure is logged, never fatal.
pub fn harden_process() {
    match platform::harden() {
        Ok(()) => {
            tracing::info!(target: thinfer_core::trace::ADAPTER, "process memory access restricted")
        }
        Err(e) => {
            tracing::warn!(target: thinfer_core::trace::ADAPTER, error = %e, "could not restrict process memory access")
        }
    }
}

#[cfg(windows)]
mod platform {
    use windows_sys::Win32::Foundation::{ERROR_SUCCESS, LocalFree};
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1, SE_KERNEL_OBJECT,
        SetSecurityInfo,
    };
    use windows_sys::Win32::Security::{
        ACL, DACL_SECURITY_INFORMATION, GetSecurityDescriptorDacl,
        PROTECTED_DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// Owner rights (`OW`, which also strips the owner's implicit right to
    /// rewrite the DACL) and SYSTEM: SYNCHRONIZE | PROCESS_QUERY_(LIMITED_)
    /// INFORMATION | PROCESS_TERMINATE only. `taskkill` / `Stop-Process` ask
    /// for query + terminate together, so query stays (metadata, not memory).
    /// Protected: nothing inherited.
    const SDDL: &str = "D:P(A;;0x101401;;;OW)(A;;0x101401;;;SY)";

    pub fn harden() -> Result<(), String> {
        let wide: Vec<u16> = SDDL.encode_utf16().chain([0]).collect();
        let mut sd: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
        // SAFETY: a NUL-terminated SDDL string and an out-param the call
        // allocates with LocalAlloc (freed below).
        let ok = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                wide.as_ptr(),
                SDDL_REVISION_1,
                &mut sd,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err("parse security descriptor".into());
        }
        let (mut present, mut defaulted) = (0, 0);
        let mut dacl: *mut ACL = std::ptr::null_mut();
        // SAFETY: `sd` is the descriptor just built; out-params are locals.
        let got = unsafe { GetSecurityDescriptorDacl(sd, &mut present, &mut dacl, &mut defaulted) };
        let res = if got == 0 || present == 0 {
            Err("descriptor has no DACL".to_string())
        } else {
            // SAFETY: the current-process pseudo handle; `dacl` points into
            // `sd`, alive until LocalFree below.
            let rc = unsafe {
                SetSecurityInfo(
                    GetCurrentProcess(),
                    SE_KERNEL_OBJECT,
                    DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    dacl,
                    std::ptr::null(),
                )
            };
            if rc == ERROR_SUCCESS {
                Ok(())
            } else {
                Err(format!("SetSecurityInfo error {rc}"))
            }
        };
        // SAFETY: `sd` came from LocalAlloc inside the conversion call.
        unsafe { LocalFree(sd) };
        res
    }
}

#[cfg(target_os = "linux")]
mod platform {
    pub fn harden() -> Result<(), String> {
        // SAFETY: plain prctl on the current process.
        if unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) } == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error().to_string())
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    pub fn harden() -> Result<(), String> {
        /// `PT_DENY_ATTACH` (sys/ptrace.h).
        const PT_DENY_ATTACH: libc::c_int = 31;
        // SAFETY: ptrace(PT_DENY_ATTACH) on the current process.
        if unsafe { libc::ptrace(PT_DENY_ATTACH, 0, std::ptr::null_mut(), 0) } == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error().to_string())
        }
    }
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
mod platform {
    pub fn harden() -> Result<(), String> {
        Err("not supported on this platform".into())
    }
}
