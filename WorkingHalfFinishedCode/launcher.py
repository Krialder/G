#!/usr/bin/env python3
"""
Gießplan Launcher - Python Environment Checker and Application Starter
Checks for Python installation and required libraries before starting the main application.
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk
import urllib.request
import webbrowser
import threading
import json
from pathlib import Path

class GiessplanLauncher:
    def __init__(self):
        self.app_directory = Path(__file__).parent
        self.config_file = self.app_directory / "launcher_config.json"
        self.config = self.load_config()
        
        # Initialize from config
        self.required_libraries = self.config.get("required_libraries", [])
        self.optional_libraries = self.config.get("optional_libraries", [])
        self.python_download_url = self.config.get("python_download_url", "https://www.python.org/downloads/")
        self.main_script = self.app_directory / self.config.get("main_script", "main.py")
        self.app_name = self.config.get("app_name", "Gießplan - Rotkreuz-Institut BBW")
        
        # Initialize GUI
        self.root = None
        self.progress_var = None
        self.status_var = None
        self.progress_bar = None
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)["launcher_config"]
            else:
                # Return default config if file doesn't exist
                return self.get_default_config()
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            "app_name": "Gießplan - Rotkreuz-Institut BBW",
            "main_script": "main.py",
            "window_size": {"width": 600, "height": 400},
            "required_libraries": ["tkinter", "json", "datetime", "re", "os", "sys", "subprocess", "atexit", "urllib", "pathlib"],
            "optional_libraries": [],
            "python_download_url": "https://www.python.org/downloads/",
            "enable_logging": True,
            "auto_install_libraries": True,
            "check_timeout": 60,
            "messages": {
                "welcome": "Willkommen zum Gießplan Launcher!\nDieser Launcher überprüft Ihr System und stellt sicher, dass alle erforderlichen Komponenten für die Gießplan-Anwendung verfügbar sind."
            }
        }
        
    def create_gui(self):
        """Create the launcher GUI"""
        self.root = tk.Tk()
        self.root.title(self.app_name)
        
        # Get window size from config
        window_config = self.config.get("window_size", {"width": 600, "height": 400})
        width, height = window_config["width"], window_config["height"]
        self.root.geometry(f"{width}x{height}")
        self.root.resizable(False, False)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text=self.app_name, 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status label
        self.status_var = tk.StringVar(value="Überprüfe Python-Installation...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                                font=("Arial", 10))
        status_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                          maximum=100, length=int(width*0.7))
        self.progress_bar.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        # Info text
        info_height = max(8, int(height/50))
        info_text = tk.Text(main_frame, height=info_height, width=int(width/10), wrap=tk.WORD)
        info_text.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Scrollbar for info text
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=info_text.yview)
        scrollbar.grid(row=3, column=2, sticky=(tk.N, tk.S))
        info_text.configure(yscrollcommand=scrollbar.set)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(10, 0))
        
        # Start button
        self.start_button = ttk.Button(button_frame, text="Gießplan starten", 
                                      command=self.start_application, state="disabled")
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        # Install Python button
        self.install_python_button = ttk.Button(button_frame, text="Python installieren", 
                                               command=self.install_python, state="disabled")
        self.install_python_button.grid(row=0, column=1, padx=(0, 10))
        
        # Exit button
        exit_button = ttk.Button(button_frame, text="Beenden", command=self.root.quit)
        exit_button.grid(row=0, column=2)
        
        self.info_text = info_text
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        return self.root
    
    def log_message(self, message):
        """Add a message to the info text widget"""
        if hasattr(self, 'info_text'):
            self.info_text.insert(tk.END, f"{message}\n")
            self.info_text.see(tk.END)
            self.root.update()
    
    def update_status(self, message):
        """Update the status label"""
        if self.status_var:
            self.status_var.set(message)
            self.root.update()
    
    def update_progress(self, value):
        """Update the progress bar"""
        if self.progress_var:
            self.progress_var.set(value)
            self.root.update()
    
    def check_python_installation(self):
        """Check if Python is installed and accessible"""
        try:
            # Check if python command is available
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.log_message(f"✅ Python gefunden: {version}")
                return True, version
            else:
                self.log_message("❌ Python nicht gefunden oder nicht im PATH")
                return False, None
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            self.log_message(f"❌ Fehler beim Überprüfen der Python-Installation: {e}")
            return False, None
    
    def check_required_libraries(self):
        """Check if all required libraries are available"""
        missing_libraries = []
        available_libraries = []
        missing_optional = []
        available_optional = []
        
        # Check required libraries
        for lib in self.required_libraries:
            try:
                __import__(lib)
                available_libraries.append(lib)
                self.log_message(f"✅ {lib} - verfügbar (erforderlich)")
            except ImportError:
                missing_libraries.append(lib)
                self.log_message(f"❌ {lib} - fehlt (erforderlich)")
        
        # Check optional libraries
        for lib in self.optional_libraries:
            try:
                __import__(lib)
                available_optional.append(lib)
                self.log_message(f"✅ {lib} - verfügbar (optional)")
            except ImportError:
                missing_optional.append(lib)
                self.log_message(f"⚠️ {lib} - fehlt (optional, für Excel-Funktionen)")
        
        return missing_libraries, available_libraries, missing_optional, available_optional
    
    def install_python(self):
        """Guide user to install Python"""
        self.update_status("Öffne Python-Download-Seite...")
        
        # Get message from config
        install_guide = self.config.get("messages", {}).get("python_install_guide", [
            "Sie werden zur Python-Download-Seite weitergeleitet.",
            "",
            "Wichtige Installationsschritte:",
            "1. Laden Sie die neueste Python-Version herunter",
            "2. Starten Sie den Installer als Administrator",
            "3. ✅ WICHTIG: Aktivieren Sie 'Add Python to PATH'",
            "4. Wählen Sie 'Install Now'",
            "5. Starten Sie nach der Installation diesen Launcher erneut"
        ])
        
        message = "\n".join(install_guide)
        messagebox.showinfo("Python Installation", message)
        webbrowser.open(self.python_download_url)
    
    def install_missing_libraries(self, missing_libraries):
        """Install missing libraries using pip"""
        if not missing_libraries:
            return True
        
        # Filter out built-in libraries that shouldn't be installed via pip
        builtin_libs = ['tkinter', 'json', 'datetime', 're', 'os', 'sys', 'subprocess', 'atexit', 'urllib', 'pathlib', 'csv']
        installable_libs = [lib for lib in missing_libraries if lib not in builtin_libs]
        
        if not installable_libs:
            self.log_message("Alle fehlenden Bibliotheken sind Teil der Standard-Python-Installation.")
            return True
        
        self.update_status("Installiere fehlende Bibliotheken...")
        
        for lib in installable_libs:
            try:
                self.log_message(f"Installiere {lib}...")
                result = subprocess.run([sys.executable, "-m", "pip", "install", lib], 
                                      capture_output=True, text=True, timeout=120)  # Increased timeout for pandas
                if result.returncode == 0:
                    self.log_message(f"✅ {lib} erfolgreich installiert")
                else:
                    self.log_message(f"❌ Fehler beim Installieren von {lib}: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                self.log_message(f"❌ Timeout beim Installieren von {lib} (>120s)")
                return False
            except Exception as e:
                self.log_message(f"❌ Unerwarteter Fehler beim Installieren von {lib}: {e}")
                return False
        
        return True
    
    def check_main_script(self):
        """Check if the main application script exists"""
        if self.main_script.exists():
            self.log_message(f"✅ Hauptanwendung gefunden: {self.main_script}")
            return True
        else:
            self.log_message(f"❌ Hauptanwendung nicht gefunden: {self.main_script}")
            return False
    
    def start_application(self):
        """Start the main Gießplan application"""
        if not self.check_main_script():
            messagebox.showerror("Fehler", "Die Hauptanwendung (main.py) wurde nicht gefunden!")
            return
        
        try:
            self.update_status("Starte Gießplan...")
            self.log_message("🚀 Starte Gießplan-Anwendung...")
            
            # Hide launcher window
            self.root.withdraw()
            
            # Start the main application
            subprocess.run([sys.executable, str(self.main_script)], 
                          cwd=str(self.app_directory))
            
        except Exception as e:
            self.log_message(f"❌ Fehler beim Starten der Anwendung: {e}")
            messagebox.showerror("Fehler", f"Konnte Gießplan nicht starten:\n{e}")
        finally:
            # Close launcher
            self.root.quit()
    
    def run_checks(self):
        """Run all system checks in a separate thread"""
        def check_thread():
            try:
                # Initialize progress
                self.update_progress(0)
                
                # Check Python installation
                self.update_status("Überprüfe Python-Installation...")
                python_available, python_version = self.check_python_installation()
                self.update_progress(25)
                
                if not python_available:
                    self.update_status("Python nicht gefunden!")
                    self.log_message("\n⚠️  Python ist nicht installiert oder nicht im PATH verfügbar.")
                    self.log_message("Klicken Sie auf 'Python installieren' um fortzufahren.")
                    self.install_python_button.config(state="normal")
                    return
                
                # Check required libraries
                self.update_status("Überprüfe erforderliche Bibliotheken...")
                missing_libs, available_libs, missing_optional, available_optional = self.check_required_libraries()
                self.update_progress(50)
                
                if missing_libs:
                    self.update_status("Installiere fehlende Bibliotheken...")
                    # Ask user permission to install missing libraries
                    response = messagebox.askyesno(
                        "Fehlende Bibliotheken", 
                        f"Die folgenden erforderlichen Bibliotheken fehlen:\n{', '.join(missing_libs)}\n\n"
                        "Sollen diese automatisch installiert werden?"
                    )
                    
                    if response:
                        if self.install_missing_libraries(missing_libs):
                            self.log_message("✅ Alle erforderlichen Bibliotheken erfolgreich installiert!")
                        else:
                            self.log_message("❌ Fehler beim Installieren der Bibliotheken.")
                            return
                    else:
                        self.log_message("⚠️  Installation abgebrochen. Die Anwendung kann möglicherweise nicht korrekt funktionieren.")
                
                # Handle optional libraries
                if missing_optional:
                    self.log_message(f"\n📝 Optionale Bibliotheken fehlen: {', '.join(missing_optional)}")
                    self.log_message("Diese werden für Excel-Funktionen benötigt.")
                    
                    response = messagebox.askyesno(
                        "Optionale Bibliotheken", 
                        f"Die folgenden optionalen Bibliotheken fehlen:\n{', '.join(missing_optional)}\n\n"
                        "Diese werden für Excel-Import/Export benötigt.\n"
                        "Die Anwendung funktioniert auch ohne diese Bibliotheken.\n\n"
                        "Sollen diese installiert werden?"
                    )
                    
                    if response:
                        if self.install_missing_libraries(missing_optional):
                            self.log_message("✅ Optionale Bibliotheken erfolgreich installiert!")
                        else:
                            self.log_message("⚠️  Fehler beim Installieren der optionalen Bibliotheken.")
                    else:
                        self.log_message("ℹ️  Optionale Bibliotheken übersprungen. Excel-Funktionen sind möglicherweise nicht verfügbar.")
                elif available_optional:
                    self.log_message(f"✅ Alle optionalen Bibliotheken verfügbar: {', '.join(available_optional)}")
                    self.log_message("Excel-Funktionen sind vollständig verfügbar.")
                
                self.update_progress(75)
                
                # Check main script
                self.update_status("Überprüfe Anwendungsdateien...")
                main_script_available = self.check_main_script()
                self.update_progress(100)
                
                # Final status
                if python_available and main_script_available:
                    self.update_status("Bereit zum Starten!")
                    self.log_message("\n🎉 Alle Überprüfungen erfolgreich! Die Anwendung kann gestartet werden.")
                    self.start_button.config(state="normal")
                else:
                    self.update_status("Fehler bei der Überprüfung!")
                    self.log_message("\n❌ Es wurden Probleme gefunden. Bitte beheben Sie diese vor dem Start.")
                
            except Exception as e:
                self.log_message(f"❌ Unerwarteter Fehler bei der Überprüfung: {e}")
                self.update_status("Fehler bei der Überprüfung!")
        
        # Start checks in background thread
        thread = threading.Thread(target=check_thread, daemon=True)
        thread.start()
    
    def run(self):
        """Run the launcher application"""
        self.create_gui()
        
        # Add welcome message from config
        welcome_message = self.config.get("messages", {}).get("welcome", 
            "Willkommen zum Gießplan Launcher!\n"
            "Dieser Launcher überprüft Ihr System und stellt sicher, dass alle "
            "erforderlichen Komponenten für die Gießplan-Anwendung verfügbar sind.\n\n"
        )
        self.log_message(welcome_message)
        
        # Start system checks after a short delay
        self.root.after(1000, self.run_checks)
        
        # Start GUI
        self.root.mainloop()

def main():
    """Main entry point for the launcher"""
    try:
        launcher = GiessplanLauncher()
        launcher.run()
    except Exception as e:
        # Fallback error handling if GUI fails
        print(f"Kritischer Fehler beim Starten des Launchers: {e}")
        input("Drücken Sie Enter zum Beenden...")

if __name__ == "__main__":
    main()
