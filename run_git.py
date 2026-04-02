import subprocess

def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        return out.decode('utf-8', errors='replace')
    except subprocess.CalledProcessError as e:
        return f"Error: Exit code {e.returncode}\nOutput: {e.output.decode('utf-8', errors='replace')}"

with open("git_log.txt", "w", encoding='utf-8') as f:
    f.write("--- STATUS ---\n")
    f.write(run("git status"))
    
    f.write("\n--- ADD ---\n")
    f.write(run("git add ."))
    
    f.write("\n--- COMMIT ---\n")
    f.write(run("git commit -m \"Add testing framework, integration tests, system performance tests, and update VISTA 2.0\""))
    
    f.write("\n--- PUSH ---\n")
    # Timeout after 20 seconds using check_output isn't easy with shell=True in py3 cleanly inline, 
    # but let's try pushing
    try:
        out = subprocess.check_output("git push", stderr=subprocess.STDOUT, shell=True, timeout=15)
        f.write(out.decode('utf-8', errors='replace'))
    except subprocess.TimeoutExpired as e:
        f.write(f"Push Timed out or waiting for credentials.")
    except subprocess.CalledProcessError as e:
        f.write(f"Push Error: {e.output.decode('utf-8', errors='replace')}")

