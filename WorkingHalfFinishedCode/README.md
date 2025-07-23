# Gießplan - Automated Plant Watering Schedule Manager

Inhaltsverzeichnis

- [Description/Overview](<#descriptionoverview>)
	- [Who is it for?](<#who-is-it-for>)
	- [Key Purpose](<#key-purpose>)
- [Features](<#features>)
	- [Core Functionality](<#core-functionality>)
	- [User Interface](<#user-interface>)
	- [Data Management](<#data-management>)
	- [Advanced Features](<#advanced-features>)
- [Installation](<#installation>)
	- [Method 1: Automated Launcher (Recommended)](<#method-1-automated-launcher-recommended>)
	- [Method 2: Manual Installation](<#method-2-manual-installation>)
- [Usage](<#usage>)
	- [Getting Started](<#getting-started>)
	- [Core Workflows](<#core-workflows>)
		- [Schedule Generation](<#schedule-generation>)
		- [Team Management](<#team-management>)
		- [Data Export](<#data-export>)
	- [Advanced Usage](<#advanced-usage>)
		- [Manual Date Management](<#manual-date-management>)
		- [CSV Table Management](<#csv-table-management>)
		- [Year Transitions](<#year-transitions>)
- [Tech Stack/Built With](<#tech-stackbuilt-with>)
	- [Core Technologies](<#core-technologies>)
	- [Standard Libraries Used](<#standard-libraries-used>)
	- [File Formats](<#file-formats>)
	- [Architecture](<#architecture>)
- [Contributing](<#contributing>)
	- [Getting Started](<#getting-started>)
	- [Contribution Guidelines](<#contribution-guidelines>)
	- [Areas for Contribution](<#areas-for-contribution>)
- [Troubleshooting/FAQ](<#troubleshootingfaq>)
	- [Installation Issues](<#installation-issues>)
		- [Python not found](<#python-not-found>)
		- [Application won't start](<#application-wont-start>)
	- [Runtime Issues](<#runtime-issues>)
		- [Multiple instances error](<#multiple-instances-error>)
		- [Permission denied for CSV export](<#permission-denied-for-csv-export>)
	- [Data Issues](<#data-issues>)
		- [Schedule generation fails](<#schedule-generation-fails>)
		- [Missing theme/styling](<#missing-themestyling>)
	- [Performance Issues](<#performance-issues>)
		- [Slow startup](<#slow-startup>)
	- [Common Questions](<#common-questions>)
		- [Q: Can I run this on Mac/Linux?](<#q-can-i-run-this-on-maclinux>)
		- [Q: How do I backup my data?](<#q-how-do-i-backup-my-data>)
		- [Q: Can I customize the schedule algorithm?](<#q-can-i-customize-the-schedule-algorithm>)
		- [Q: How do I add holidays or exceptions?](<#q-how-do-i-add-holidays-or-exceptions>)
		- [Q: Can multiple people use this simultaneously?](<#q-can-multiple-people-use-this-simultaneously>)
	- [Getting Help](<#getting-help>)
	- [System Requirements](<#system-requirements>)
		- [Minimum Requirements](<#minimum-requirements>)
		- [Recommended Requirements](<#recommended-requirements>)

## Description/Overview

**Gießplan** is a comprehensive Python application designed for the **Rotkreuz-Institut BBW** to manage fair and automated plant watering schedules. The application automatically generates balanced assignments for multiple people, ensuring everyone shares the watering responsibilities equally over time.

### Who is it for?

- **Administrators/Managers**: Generate and manage watering schedules
- **Team Members**: View assignments and track watering duties

### Key Purpose

Transform manual watering schedule coordination into an automated, fair, and transparent system that saves time and prevents conflicts.

##  Features

###  **Core Functionality**

- **Intelligent Schedule Generation**: Creates 6-week watering schedules using weighted selection algorithms
- **Fair Assignment System**: Automatically balances workload among all participants
- **Multi-Person Management**: Add/remove people with automatic weight rebalancing
- **Year Transition Support**: Seamlessly handles year transitions with separate data files

###  **User Interface**

- **GUI**: Intuitive tkinter-based interface
- **Real-time Updates**: Instant feedback and live schedule previews
- **Manual Override**: Add or remove specific dates/weeks manually
- **Status Indicators**: Visual cues for current, upcoming, and past assignments

###  **Data Management**

- **Persistent Storage**: JSON-based data storage for people and history
- **Excel Export**: Automated generation of Excel files with statistics
- **CSV Export**: User-friendly CSV tables for team distribution
- **Backup System**: Automatic data backup and recovery functionality

###  **Advanced Features**

- **Smart Launcher**: Automatic Python and dependency checking
- **Theme Integration**: dark theme with Red Cross Institute branding
- **Table Management**: CSV generation for end-user consumption
- **Single Instance Control**: Prevents multiple application instances

##  Installation

### Method 1: Automated Launcher (Recommended)

1. **Download** all project files to a folder
2. **Create Desktop Shortcut**:

   ```batch

   # Double-click this file to create a desktop icon

   create_desktop_shortcut.bat

   ```

3. **Start Application**: Use the desktop icon or double-click `start_giessplan.bat`

The launcher will automatically:

- ✅ Check for Python installation
- ✅ Verify required libraries
- ✅ Install missing components (with permission)
- ✅ Start the application

### Method 2: Manual Installation

1. **Install Python 3.7+**: Download from [python.org](https://python.org)

   - ⚠️ **Important**: Check "Add Python to PATH" during installation

2. **Install Dependencies**:

```bash

   # Install only if you need Excel export features

   pip install openpyxl    # Excel file creation and editing

   pip install pandas      # Advanced data manipulation (Excel statistics)

   ```

   **Note**: The application works fully without optional libraries. Excel features will be disabled if these libraries are not available.

   ```

  

3. **Run Application**:

   ```

   python main.py

   ```

  

### Method 3: PowerShell (Advanced Users)

  

```powershell

# Normal start with system checks

.\start_giessplan.ps1

  

# Verbose output for troubleshooting

.\start_giessplan.ps1 -Verbose

  

# Force start without checks

.\start_giessplan.ps1 -Force

```

##  Usage

### Getting Started

1. **Launch** the application using one of the installation methods
2. **Add People**: Enter names and click "Add Person"
3. **Generate Schedule**: Click "Generate Gießplan" for a 6-week schedule
4. **Review & Adjust**: Use manual controls to modify specific assignments
5. **Export**: Schedules automatically save to Excel and CSV formats

### Core Workflows

####  **Schedule Generation**

```

Add People → Generate Schedule → Review → Export → Distribute

```

####  **Team Management**

```

Current Team → Add/Remove Members → Rebalance Weights → Update Schedule

```

####  **Data Export**

```

Generated Schedule → Excel Export → CSV Table → Team Distribution

```

### Advanced Usage

#### **Manual Date Management**

- Add specific dates using the date picker
- Remove assignments for holidays or absences
- Override automatic assignments when needed

#### **CSV Table Management**

- Generate user-friendly CSV files for team consumption
- Professional formatting with status indicators
- Read-only protection to prevent accidental changes

#### **Year Transitions**

- Automatic year detection and file management
- Seamless data migration between years
- Historical data preservation

##  Tech Stack/Built With

### **Core Technologies**

- **Python 3.7+**: Main programming language
- **Tkinter**: GUI framework (built-in with Python)
- **JSON**: Data storage and configuration
- **subprocess**: System integration and process management

### **Standard Libraries Used**

- `datetime` - Date and time handling
- `json` - Data serialization
- `os` & `sys` - Operating system interface
- `re` - Regular expressions
- `pathlib` - Modern path handling
- `threading` - Background task processing
- `urllib` - URL and web handling

### **File Formats**

- **JSON**: Primary data storage (`.json`)
- **Excel**: Schedule exports (`.xlsx`)
- **CSV**: User-friendly tables (`.csv`)
- **Batch/PowerShell**: Launch scripts (`.bat`, `.ps1`)

### **Architecture**

- **Modular Design**: Separate modules for GUI, data, scheduling, and management
- **Theme System**: Extensible theming with fallback support
- **Launcher System**: Environment validation and automated setup
- **Plugin Architecture**: Easy integration of new features

## Contributing

We welcome contributions to improve the Gießplan application! Here's how you can help:

### **Getting Started**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Test thoroughly using the launcher system
5. Commit your changes (`git commit -m 'Add AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### **Contribution Guidelines**

- Follow Python PEP 8 style guidelines
- Add documentation for new features
- Include tests for new functionality
- Update README.md if needed
- Respect the existing code structure

### **Areas for Contribution**

- 🐛 Bug fixes and error handling
- 🎨 UI/UX improvements
- 📱 Mobile app development
- 🔔 Notification systems
- 📊 Advanced analytics
- 🌐 Internationalization

## Troubleshooting/FAQ

### **Installation Issues**

####  **Python not found**

```

Error: Python nicht gefunden oder nicht im PATH

```

**Solution:**

1. Download Python from [python.org](https://python.org)
2. ✅ **Important**: Check "Add Python to PATH" during installation
3. Restart computer
4. Run launcher again

####  **Application won't start**

```

Error: Hauptanwendung nicht gefunden: main.py

```

**Solution:**

- Ensure all files are in the same directory
- Don't rename or move `main.py`
- Re-download project files if necessary

### **Runtime Issues**

####  **Multiple instances error**

```

Error: Eine andere Instanz von Gießplan läuft bereits

```

**Solution:**

- Close other instances of the application
- Check Windows Task Manager for hidden processes
- Restart computer if process is stuck

####  **Permission denied for CSV export**

```

Error: Permission denied when creating CSV

```

**Solution:**

- Close Excel or other programs using the CSV file
- Run application as Administrator
- Check file/folder permissions

### **Data Issues**

####  **Schedule generation fails**

```

Error: Cannot generate schedule

```

**Solution:**

- Ensure at least 2 people are added
- Check that data files aren't corrupted
- Use backup recovery if available

####  **Missing theme/styling**

```

Warning: Theme integration failed

```

**Solution:**

- Application works with basic styling
- Check if `theme_integration.py` exists
- Restart application

### **Performance Issues**

####  **Slow startup**

**Possible Causes:**

- Antivirus scanning Python files
- Large data files
- Network drive location

**Solutions:**

- Add Python to antivirus exceptions
- Move application to local drive
- Close unnecessary programs

### **Common Questions**

#### **Q: Can I run this on Mac/Linux?**

A: The core application works on all platforms, but launcher scripts are Windows-specific. Use `python main.py` directly on other platforms.

#### **Q: How do I backup my data?**

A: Data is stored in JSON files (`people_YYYY.json`). Copy these files to backup your schedules and history.

#### **Q: Can I customize the schedule algorithm?**

A: Yes, modify the weight calculation in `schedule.py`. The system uses configurable parameters.

#### **Q: How do I add holidays or exceptions?**

A: Use the manual date management features to remove assignments for specific dates or weeks.

#### **Q: Can multiple people use this simultaneously?**

A: No, the application prevents multiple instances. Use the CSV export feature to share schedules with team members.

### **Getting Help**

1. **Check Logs**: Look at console output for detailed error messages
2. **Restart**: Try restarting the application
3. **Reinstall**: Use the launcher to verify all components
4. **Contact Support**: Reach out to your IT department with specific error messages

### **System Requirements**

#### **Minimum Requirements**

- Windows 7 or newer (windows 7 is save kappa)
- Python 3.7+
- 50 MB free disk space
- 512 MB RAM

#### **Recommended Requirements**

- Windows 11
- Python 3.9+
- 100 MB free disk space
- 1 GB RAM
- Internet connection (for initial Python installation)

For additional support, consult the launcher documentation (`LAUNCHER_README.md`) and table management guide (`TABELLE_MANAGEMENT_DOCS.md`).
