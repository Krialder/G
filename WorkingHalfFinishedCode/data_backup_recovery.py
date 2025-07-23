#!/usr/bin/env python3
"""
Data Backup and Recovery System for Watering Schedule Management

This module provides functions to:
1. Create and maintain people.json as a master template
2. Backup current year data to people.json 
3. Recover/reinstall data from people.json
4. Migrate data between installations
"""

import os
import json
import datetime
import shutil
from tkinter import messagebox
import data

def create_master_template():
    """Create or update people.json as master template with current year's balanced data"""
    current_year = data.get_current_year()
    
    # Use current data as template
    template_data = {
        "PEOPLE": data.PEOPLE.copy(),
        "WEIGHTS": data.WEIGHTS.copy(),
        "EXTRA_WEIGHTS": data.EXTRA_WEIGHTS.copy(),
        "EXPERIENCE_OVERRIDES": data.experience_overrides.copy(),
        "WATERING_HISTORY": {person: [] for person in data.PEOPLE},  # Empty for template
        "METADATA": {
            "created_date": datetime.datetime.now().isoformat(),
            "source_year": current_year,
            "total_people": len(data.PEOPLE),
            "description": "Master template created from balanced system data",
            "version": "2.0"
        }
    }
    
    try:
        with open("people.json", "w", encoding='utf-8') as file:
            json.dump(template_data, file, indent=2, ensure_ascii=False)
        print(f"✅ Created master template people.json from {current_year} data")
        print(f"   📊 {len(data.PEOPLE)} people with balanced weights: {data.WEIGHTS}")
        return True
    except Exception as e:
        print(f"❌ Failed to create master template: {e}")
        return False

def backup_current_year_to_template():
    """Backup current year's data to people.json (preserving balanced weights)"""
    current_year = data.get_current_year()
    
    # Ask user for confirmation
    backup_msg = f"""
Backup current {current_year} data to people.json template?

This will:
✅ Save your current {len(data.PEOPLE)} people
✅ Save balanced weights: {data.WEIGHTS[:5]}{'...' if len(data.WEIGHTS) > 5 else ''}
✅ Save experience levels: {len(data.experience_overrides)} manual overrides
✅ Create empty watering history (for clean template)

This becomes your master template for recovery/reinstallation.
"""
    
    if messagebox.askyesno("Backup to Template", backup_msg):
        if create_master_template():
            messagebox.showinfo("Backup Complete", 
                f"✅ Successfully backed up {current_year} data to people.json\n\n"
                f"This template can now be used for:\n"
                f"• Data recovery\n"
                f"• Code reinstallation\n" 
                f"• New year initialization")
            return True
        else:
            messagebox.showerror("Backup Failed", "❌ Failed to create backup template")
            return False
    return False

def restore_from_template(target_year=None):
    """Restore data from people.json template"""
    if not os.path.exists("people.json"):
        print("❌ No people.json template found")
        return False
    
    if target_year is None:
        target_year = data.get_current_year()
    
    try:
        # Load template
        with open("people.json", "r", encoding='utf-8') as file:
            template_data = json.load(file)
        
        # Apply template data
        data.PEOPLE.clear()
        data.PEOPLE.extend(template_data.get("PEOPLE", []))
        data.WEIGHTS.clear()
        data.WEIGHTS.extend(template_data.get("WEIGHTS", []))
        data.EXTRA_WEIGHTS.clear()
        data.EXTRA_WEIGHTS.extend(template_data.get("EXTRA_WEIGHTS", []))
        data.experience_overrides.clear()
        data.experience_overrides.update(template_data.get("EXPERIENCE_OVERRIDES", {}))
        
        # Start with empty watering history for target year
        data.watering_history.clear()
        data.watering_history.update({person: [] for person in data.PEOPLE})
        
        # Set file path and save
        data.FILE_PATH = f"people_{target_year}.json"
        data.save_to_file()
        
        metadata = template_data.get("METADATA", {})
        source_year = metadata.get("source_year", "unknown")
        
        print(f"✅ Restored data from people.json template to {target_year}")
        print(f"   📊 {len(data.PEOPLE)} people restored")
        print(f"   ⚖️  Balanced weights from year {source_year}: {data.WEIGHTS}")
        print(f"   🧠 {len(data.experience_overrides)} experience overrides restored")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to restore from template: {e}")
        return False

def migrate_data_package():
    """Create a complete data migration package"""
    current_year = data.get_current_year()
    package_name = f"watering_data_backup_{current_year}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Create backup directory
        if not os.path.exists("backups"):
            os.makedirs("backups")
        
        package_dir = os.path.join("backups", package_name)
        os.makedirs(package_dir)
        
        # Copy all year files
        year_files_copied = 0
        for file in os.listdir("."):
            if file.startswith("people_") and file.endswith(".json"):
                shutil.copy2(file, package_dir)
                year_files_copied += 1
        
        # Copy people.json if it exists
        if os.path.exists("people.json"):
            shutil.copy2("people.json", package_dir)
        
        # Create master template if it doesn't exist
        if not os.path.exists("people.json"):
            create_master_template()
            shutil.copy2("people.json", package_dir)
        
        # Create migration info file
        migration_info = {
            "created_date": datetime.datetime.now().isoformat(),
            "source_year": current_year,
            "year_files_included": year_files_copied,
            "current_people": data.PEOPLE.copy(),
            "current_weights": data.WEIGHTS.copy(),
            "migration_instructions": [
                "1. Copy all JSON files to your new installation directory",
                "2. Run 'python data_backup_recovery.py --restore' to load template",
                "3. Use GUI to switch between years as needed",
                "4. Your balanced weights and experience levels will be preserved"
            ]
        }
        
        with open(os.path.join(package_dir, "migration_info.json"), "w", encoding='utf-8') as file:
            json.dump(migration_info, file, indent=2, ensure_ascii=False)
        
        print(f"✅ Created migration package: {package_dir}")
        print(f"   📁 {year_files_copied} year files included")
        print(f"   📋 Master template included")
        print(f"   📝 Migration instructions included")
        
        return package_dir
        
    except Exception as e:
        print(f"❌ Failed to create migration package: {e}")
        return None

def install_from_migration_package(package_dir):
    """Install data from a migration package"""
    if not os.path.exists(package_dir):
        print(f"❌ Migration package not found: {package_dir}")
        return False
    
    try:
        # Copy all JSON files
        files_installed = 0
        for file in os.listdir(package_dir):
            if file.endswith(".json") and file != "migration_info.json":
                shutil.copy2(os.path.join(package_dir, file), ".")
                files_installed += 1
        
        # Load migration info
        migration_info_path = os.path.join(package_dir, "migration_info.json")
        if os.path.exists(migration_info_path):
            with open(migration_info_path, "r", encoding='utf-8') as file:
                migration_info = json.load(file)
            source_year = migration_info.get("source_year")
            print(f"📦 Installing from package created for year {source_year}")
        
        # Restore from template
        if restore_from_template():
            print(f"✅ Successfully installed {files_installed} data files")
            print(f"   🔄 Data restored from people.json template")
            return True
        else:
            print(f"⚠️  Files copied but template restoration failed")
            return False
            
    except Exception as e:
        print(f"❌ Failed to install from migration package: {e}")
        return False

def check_data_integrity():
    """Check integrity of current data and suggest repairs"""
    print("🔍 Checking Data Integrity...")
    
    issues = []
    suggestions = []
    
    # Check basic data consistency
    if len(data.WEIGHTS) != len(data.PEOPLE):
        issues.append(f"WEIGHTS length ({len(data.WEIGHTS)}) != PEOPLE length ({len(data.PEOPLE)})")
        suggestions.append("Run data.refresh_dependencies() to fix")
    
    if len(data.EXTRA_WEIGHTS) != len(data.PEOPLE):
        issues.append(f"EXTRA_WEIGHTS length ({len(data.EXTRA_WEIGHTS)}) != PEOPLE length ({len(data.PEOPLE)})")
        suggestions.append("Run data.refresh_dependencies() to fix")
    
    # Check for missing watering history entries
    missing_history = [person for person in data.PEOPLE if person not in data.watering_history]
    if missing_history:
        issues.append(f"Missing watering history for: {missing_history}")
        suggestions.append("Run data.refresh_dependencies() to fix")
    
    # Check for orphaned watering history
    orphaned_history = [person for person in data.watering_history if person not in data.PEOPLE]
    if orphaned_history:
        issues.append(f"Orphaned watering history for: {orphaned_history}")
        suggestions.append("Run data.refresh_dependencies() to fix")
    
    # Check weight distribution
    if data.WEIGHTS:
        min_weight = min(data.WEIGHTS)
        max_weight = max(data.WEIGHTS)
        weight_ratio = max_weight / min_weight if min_weight > 0 else float('inf')
        
        if weight_ratio > 10:
            issues.append(f"Extreme weight imbalance: {min_weight}-{max_weight} (ratio: {weight_ratio:.1f})")
            suggestions.append("Run data.normalize_extreme_weights() to balance")
    
    # Check template availability
    if not os.path.exists("people.json"):
        issues.append("No people.json template available for recovery")
        suggestions.append("Run create_master_template() to create backup")
    
    # Report results
    if not issues:
        print("✅ Data integrity check passed - no issues found")
        return True
    else:
        print(f"⚠️  Found {len(issues)} data integrity issues:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 Suggested fixes:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        
        return False

def emergency_data_recovery():
    """Emergency recovery function when data is corrupted"""
    print("🚨 Emergency Data Recovery Mode")
    
    recovery_options = []
    
    # Check for people.json template
    if os.path.exists("people.json"):
        recovery_options.append(("Template", "Restore from people.json template"))
    
    # Check for recent year files
    current_year = datetime.date.today().year
    for year in range(current_year - 2, current_year + 1):
        year_file = f"people_{year}.json"
        if os.path.exists(year_file):
            recovery_options.append((f"Year {year}", f"Restore from {year_file}"))
    
    # Check for backup packages
    if os.path.exists("backups"):
        backup_dirs = [d for d in os.listdir("backups") if d.startswith("watering_data_backup_")]
        if backup_dirs:
            latest_backup = sorted(backup_dirs)[-1]
            recovery_options.append(("Backup", f"Restore from latest backup: {latest_backup}"))
    
    if not recovery_options:
        print("❌ No recovery options available")
        print("💡 You may need to manually recreate your people data")
        return False
    
    print(f"🔧 Found {len(recovery_options)} recovery options:")
    for i, (name, description) in enumerate(recovery_options, 1):
        print(f"   {i}. {name}: {description}")
    
    # For automation, try template first
    if any(option[0] == "Template" for option in recovery_options):
        print("\n🔄 Attempting template recovery...")
        if restore_from_template():
            print("✅ Emergency recovery successful using template")
            return True
    
    print("⚠️  Automatic recovery options available - manual intervention may be needed")
    return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "--backup":
            create_master_template()
        elif command == "--restore":
            restore_from_template()
        elif command == "--check":
            check_data_integrity()
        elif command == "--emergency":
            emergency_data_recovery()
        elif command == "--migrate":
            package = migrate_data_package()
            if package:
                print(f"\n📋 To use this migration package:")
                print(f"   1. Copy folder: {package}")
                print(f"   2. On new installation: python data_backup_recovery.py --install-package {package}")
        elif command == "--install-package" and len(sys.argv) > 2:
            install_from_migration_package(sys.argv[2])
        else:
            print("Available commands:")
            print("  --backup     Create/update people.json template")
            print("  --restore    Restore from people.json template")
            print("  --check      Check data integrity")
            print("  --emergency  Emergency data recovery")
            print("  --migrate    Create migration package")
            print("  --install-package <dir>  Install from migration package")
    else:
        print("🛠️  Data Backup and Recovery System")
        print("=" * 40)
        
        # Check current state
        check_data_integrity()
        
        print("\n🔧 Available Actions:")
        print("1. Create master template from current data")
        print("2. Check if people.json template exists")
        
        if os.path.exists("people.json"):
            print("   ✅ people.json template exists")
            try:
                with open("people.json", "r") as f:
                    template = json.load(f)
                metadata = template.get("METADATA", {})
                print(f"   📊 Template has {len(template.get('PEOPLE', []))} people")
                if "source_year" in metadata:
                    print(f"   📅 Created from year {metadata['source_year']}")
            except:
                print("   ⚠️  Template exists but may be corrupted")
        else:
            print("   ❌ No people.json template found")
            print("   💡 Run with --backup to create one")
