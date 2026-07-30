param(
    [ValidatePattern("^[a-z_][a-z0-9_-]*$")]
    [string]$LinuxUser
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL is not available. Enable the Windows Subsystem for Linux first."
}

$installed = wsl.exe --list --quiet
if ($installed -contains "Ubuntu-24.04") {
    Write-Host "Ubuntu-24.04 is already installed."
} else {
    Write-Host "Installing Ubuntu-24.04 for JitPi05..."
    wsl.exe --install --distribution Ubuntu-24.04 --no-launch
    Write-Host "Ubuntu is installed. Launch it once to create your Linux user."
}

if ($LinuxUser) {
    # Windows PowerShell 5 promotes native stderr (including WSL's harmless
    # localhost-proxy warning) to an ErrorRecord. Temporarily avoid treating
    # that warning as a terminating PowerShell error and rely on exit codes.
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        wsl.exe -d Ubuntu-24.04 -u root -- id $LinuxUser 2>$null
        $userExists = $LASTEXITCODE -eq 0

        if (-not $userExists) {
            Write-Host "Creating Linux user '$LinuxUser'. Choose its password when prompted."
            wsl.exe -d Ubuntu-24.04 -u root -- adduser $LinuxUser
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create Linux user '$LinuxUser'."
            }
        }

        wsl.exe -d Ubuntu-24.04 -u root -- usermod -aG sudo $LinuxUser 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to add Linux user '$LinuxUser' to the sudo group."
        }

        # Write inside Linux so Windows PowerShell cannot encode redirected
        # text as UTF-16. LinuxUser is constrained by ValidatePattern above.
        $configureDefaultUser = "printf '[user]\ndefault=$LinuxUser\n' > /etc/wsl.conf"
        wsl.exe -d Ubuntu-24.04 -u root -- sh -c $configureDefaultUser 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to configure '$LinuxUser' as the default WSL user."
        }

        wsl.exe --terminate Ubuntu-24.04 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "The user was configured, but Ubuntu-24.04 could not be restarted."
        }
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }

    Write-Host "Default WSL user configured: $LinuxUser"
} else {
    Write-Host "No Linux user was configured."
    Write-Host "Re-run this script with -LinuxUser <name> to create one securely."
}

Write-Host "Next: clone the repository under ~/projects inside Ubuntu,"
Write-Host "then run: bash scripts/bootstrap_ubuntu.sh"
