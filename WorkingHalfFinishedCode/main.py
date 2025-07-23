# Entry point for the Gießplan application

import os
import sys
import atexit
import tkinter.messagebox as messagebox

def check_single_instance():
    """Ensure only one instance of the application is running"""
    pid_file = "giessplan.pid"
    
    # Check if PID file exists
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                old_pid = int(f.read().strip())
            
            # Check if process is still running (Windows compatible)
            try:
                if os.name == 'nt':  # Windows
                    import subprocess
                    result = subprocess.run(['tasklist', '/fi', f'PID eq {old_pid}'], 
                                          capture_output=True, text=True, encoding='utf-8', errors='ignore')
                    if result.stdout and str(old_pid) in result.stdout:
                        # Process exists
                        messagebox.showerror("Gießplan bereits geöffnet", 
                                           "Eine andere Instanz von Gießplan läuft bereits.\n\nBitte schließen Sie die andere Anwendung zuerst.")
                        sys.exit(1)
                else:  # Unix/Linux/Mac
                    os.kill(old_pid, 0)  # Signal 0 just checks if process exists
                    messagebox.showerror("Gießplan already running", 
                                       "Another instance of Gießplan is already running.\n\nPlease close the other application first.")
                    sys.exit(1)
            except (subprocess.CalledProcessError, OSError, FileNotFoundError):
                # Process doesn't exist, remove stale PID file
                try:
                    os.remove(pid_file)
                except:
                    pass
        except (ValueError, FileNotFoundError):
            # Invalid or missing PID file, remove it
            try:
                os.remove(pid_file)
            except:
                pass
    
    # Create new PID file
    try:
        with open(pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Clean up PID file on exit
        atexit.register(lambda: os.path.exists(pid_file) and os.remove(pid_file))
    except Exception as e:
        print(f"Warning: Could not create PID file: {e}")

def main():
    """Main entry point for the Gießplan application"""
    try:
        # Check for single instance before importing GUI
        check_single_instance()
        
        print("🚀 Starting Gießplan application...")
        
        # Import and start GUI
        from gui import root
        
        print("✅ Gießplan started successfully")
        print("📍 Application is now running...")
        
        # The application starts when gui.py is imported and root.mainloop() is called there.
        
    except Exception as e:
        print(f"❌ Critical error starting Gießplan: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to show error message to user if possible
        try:
            messagebox.showerror("Kritischer Fehler", 
                               f"Gießplan konnte nicht gestartet werden:\n\n{e}\n\n"
                               "Überprüfen Sie die Konsole für weitere Details.")
        except:
            pass  # GUI might not be available
        
        sys.exit(1)

if __name__ == "__main__":
    main()