import tkinter as tk
from tkinter import messagebox, ttk
import data
from data import save_to_file, refresh_dependencies, add_new_person_with_context, remove_person_and_rebalance, reload_current_data, get_available_years, load_year_data, get_current_year, get_week_data, get_week_data_with_ersatz, update_week_data, update_week_data_with_ersatz, get_person_experience_level, set_person_experience_level, remove_person_experience_override, get_all_experience_levels, analyze_watering_imbalance, balance_watering_history, get_watering_history_report
from schedule import show_schedule
from tabelle_management import TabelleManager
import datetime
import re

# Import backup recovery system
try:
    import data_backup_recovery
    BACKUP_AVAILABLE = True
    print("✅ Data backup recovery system imported successfully")
except ImportError as e:
    BACKUP_AVAILABLE = False
    print(f"⚠️  Data backup recovery system not available: {e}")

# Try to import theme integration, fallback to basic styling if not available
try:
    from theme_integration import apply_rki_theme_to_app, RKIModernColors
    THEME_AVAILABLE = True
    print("✅ Theme integration imported successfully")
except ImportError as e:
    THEME_AVAILABLE = False
    print(f"❌ Theme integration failed: {e}")

# Create the GUI
root = tk.Tk()
root.title("Gießplan Generator - Rotkreuz-Institut BBW")
root.geometry("1200x800")  # Increased size for modern layout

# Apply theme
if THEME_AVAILABLE:
    print("🎨 Applying modern theme...")
    # Apply Red Cross Institute modern dark theme
    theme, widgets = apply_rki_theme_to_app(root, modern_theme=True)
    theme_instance = theme  # Make theme_instance available
    colors = RKIModernColors()
    print("✅ Modern theme applied successfully")
    
    # Add compatibility mappings for legacy color names
    colors.LIGHT_GRAY = colors.CARD_BACKGROUND
    colors.RED_CROSS_WHITE = colors.DARK_BACKGROUND
    colors.PALE_BLUE = colors.SOFT_BLUE
    colors.PROFESSIONAL_BLUE = colors.PRIMARY_TEXT
    colors.LIGHT_ORANGE = colors.WARNING_ORANGE
    colors.WARM_ORANGE = colors.WARNING_ORANGE
    colors.DARK_GRAY = colors.BORDER_COLOR
    colors.MEDIUM_GRAY = colors.SECONDARY_TEXT
    colors.ELECTRIC_BLUE = colors.SOFT_BLUE
    colors.BRIGHT_BLUE = colors.LIGHT_BLUE
    colors.MUTED_BLUE = colors.MUTED_BLUE
else:
    # Fallback to basic theme
    theme_instance = theme_integration.RKITheme(root, modern_theme=False)
    theme_instance.apply_theme()
    widgets = theme_integration.create_styled_widgets(modern_theme=False)
    colors = theme_instance.colors

# Create main container with professional spacing
main_frame = widgets['frame'](root, padding="20")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Configure grid weights
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(2, weight=1)

# Title
title_label = widgets['title_label'](main_frame, text="🌱 Gießplan Generator")
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

# Year selection frame
year_frame = widgets['labelframe'](main_frame, text="📅 Year Selection", padding="10")
year_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
year_frame.columnconfigure(1, weight=1)

# Year selection widgets
widgets['label'](year_frame, text="Select Year:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

year_var = tk.StringVar()
year_combobox = widgets['combobox'](year_frame, textvariable=year_var, state="readonly", width=10)
year_combobox.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))

def refresh_years():
    """Refresh the list of available years"""
    available_years = get_available_years()
    year_combobox['values'] = available_years
    
    # Get the current year being worked on
    current_year = get_current_year()
    
    # Set the combobox to show the current year
    if current_year in available_years:
        year_var.set(current_year)
    elif available_years:
        # If current year not in available years, set to the most recent
        year_var.set(available_years[-1])
    
    update_status()

def on_year_changed(event=None):
    """Handle year selection change"""
    try:
        selected_year = int(year_var.get())
        if load_year_data(selected_year):
            # Update all displays with the new year data
            update_all_displays()
            update_status()
            # Update tabelle manager for new year
            if 'tabelle_manager' in globals():
                tabelle_manager.update_csv_file_path()
                tabelle_manager.update_displays()
            messagebox.showinfo("Success", f"Switched to year {selected_year}")
        else:
            messagebox.showerror("Error", f"No data file found for year {selected_year}")
            # Reset to current year if failed
            refresh_years()
    except ValueError:
        messagebox.showerror("Error", "Invalid year selected")
        refresh_years()

year_combobox.bind('<<ComboboxSelected>>', on_year_changed)

# Refresh years button
refresh_years_btn = widgets['small_button'](year_frame, text="🔄 Refresh Years", command=refresh_years)
refresh_years_btn.grid(row=0, column=2, padx=(10, 0))

# Current year info
current_year_label = widgets['label'](year_frame, text="")
current_year_label.grid(row=0, column=3, padx=(20, 0))

# Status bar - moved inside the Year Selection container, but leave space for theme switch
status_var = tk.StringVar()
status_bar = widgets['label'](year_frame, textvariable=status_var)
status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

# Theme variable
theme_var = tk.BooleanVar(value=True)  # True = Dark mode, False = Light mode

# Create theme switch directly here - positioned below "Working on:"
print("🔧 Creating theme switch directly...")
theme_frame = widgets['frame'](year_frame)
theme_frame.grid(row=1, column=3, sticky='w', padx=(20, 0), pady=(5, 0))

# Theme switch components
widgets['label'](theme_frame, text="🌙", font=('Segoe UI', 10)).grid(row=0, column=0, padx=(0, 3))

switch_frame = widgets['frame'](theme_frame)
switch_frame.grid(row=0, column=1, padx=2)

# Switch background (acts like the track)
switch_bg = tk.Frame(switch_frame, width=35, height=18, relief='solid', bd=1)
switch_bg.grid(row=0, column=0)
switch_bg.grid_propagate(False)

# Switch handle (the moving part) - define toggle_theme later
switch_handle = tk.Button(switch_bg, text="", width=1, height=1, relief='raised', bd=1)
switch_handle.place(x=2, y=2, width=14, height=14)

widgets['label'](theme_frame, text="☀️", font=('Segoe UI', 10)).grid(row=0, column=2, padx=(3, 0))

print("✅ Theme switch created successfully")

def update_theme_switch_appearance():
    """Update the visual appearance of the theme switch"""
    try:
        if THEME_AVAILABLE:
            colors = RKIModernColors() if theme_var.get() else None
        else:
            colors = None
            
        if theme_var.get():  # Dark mode
            # Handle on the left (dark mode active)
            switch_handle.place(x=2, y=2, width=16, height=16)
            if colors:
                switch_bg.config(bg=colors.CARD_BACKGROUND if hasattr(colors, 'CARD_BACKGROUND') else '#36393f')
                switch_handle.config(bg=colors.RED_CROSS_RED if hasattr(colors, 'RED_CROSS_RED') else '#DC143C')
            else:
                switch_bg.config(bg='#36393f')
                switch_handle.config(bg='#DC143C')
        else:  # Light mode  
            # Handle on the right (light mode active)
            switch_handle.place(x=22, y=2, width=16, height=16)
            switch_bg.config(bg='#f1f3f5')
            switch_handle.config(bg='#3182ce')
    except Exception as e:
        print(f"⚠️  Error updating switch appearance: {e}")
        # Fallback to simple appearance
        if theme_var.get():
            switch_handle.place(x=2, y=2, width=16, height=16)
            switch_bg.config(bg='gray')
            switch_handle.config(bg='red')
        else:
            switch_handle.place(x=22, y=2, width=16, height=16) 
            switch_bg.config(bg='lightgray')
            switch_handle.config(bg='blue')

def toggle_theme():
    """Toggle between dark and light themes"""
    global widgets  # Make sure we can update the global widgets
    
    # Toggle the state
    theme_var.set(not theme_var.get())
    is_dark_mode = theme_var.get()
    
    print(f"🔄 Toggling theme: {'Dark' if is_dark_mode else 'Light'} mode")
    
    try:
        # Store reference to current switch widgets before theme change
        current_switch_handle = switch_handle
        current_switch_bg = switch_bg
        current_theme_frame = theme_frame
        
        if is_dark_mode:
            # Switch to dark mode
            print("🌙 Switching to dark mode...")
            if THEME_AVAILABLE:
                apply_rki_theme_to_app(root, modern_theme=True, light_mode=False)
            else:
                print("⚠️  Dark theme not available, using default")
        else:
            # Switch to light mode
            print("☀️ Switching to light mode...")
            if THEME_AVAILABLE:
                apply_rki_theme_to_app(root, modern_theme=True, light_mode=True)
            else:
                print("⚠️  Light theme not available")
                messagebox.showerror("Theme Error", "Light theme not available.")
                theme_var.set(True)  # Reset to dark mode
                update_theme_switch_appearance()
                return
        
        # Update switch appearance using stored references
        try:
            update_theme_switch_appearance()
        except:
            # If update fails, the references might be invalid
            # Set up the theme switch again
            setup_theme_switch()
        
        # Force refresh all widgets to apply new theme
        root.update_idletasks()
        print(f"✅ Theme switched to {'dark' if is_dark_mode else 'light'} mode")
        
    except Exception as e:
        print(f"❌ Theme switch failed: {e}")
        messagebox.showerror("Theme Error", f"Failed to switch theme: {str(e)}")
        # Reset to previous state
        theme_var.set(not theme_var.get())
        try:
            update_theme_switch_appearance()
        except:
            setup_theme_switch()

# Configure the switch handle command now that toggle_theme is defined
switch_handle.config(command=lambda: toggle_theme())
print("✅ Theme switch command configured")

# Initialize theme switch appearance
update_theme_switch_appearance()
print("✅ Theme switch appearance initialized")

def setup_theme_switch():
    """Set up or re-setup the theme switch in the UI"""
    global switch_handle, switch_bg, theme_frame
    
    try:
        # Find the year_frame container 
        container = None
        for child in root.winfo_children():
            if hasattr(child, 'grid_info') and child.grid_info().get('row') == 1:
                container = child
                break
        
        if not container:
            print("⚠️  Could not find container for theme switch")
            return
            
        # Create theme toggle frame - place it under "Working on:" in row 1, column 3
        theme_frame = widgets['frame'](container)
        theme_frame.grid(row=1, column=3, sticky='w', padx=(20, 0), pady=(5, 0))

        # Theme switch components
        widgets['label'](theme_frame, text="🌙", font=('Segoe UI', 10)).grid(row=0, column=0, padx=(0, 3))

        switch_frame = widgets['frame'](theme_frame)
        switch_frame.grid(row=0, column=1, padx=2)

        # Switch background (acts like the track)
        switch_bg = tk.Frame(switch_frame, width=35, height=18, relief='solid', bd=1)
        switch_bg.grid(row=0, column=0)
        switch_bg.grid_propagate(False)

        # Switch handle (the moving part)
        switch_handle = tk.Button(switch_bg, text="", width=1, height=1, relief='raised', bd=1,
                                 command=lambda: toggle_theme())
        switch_handle.place(x=2, y=2, width=14, height=14)

        widgets['label'](theme_frame, text="☀️", font=('Segoe UI', 10)).grid(row=0, column=2, padx=(3, 0))

        # Update appearance to match current theme
        update_theme_switch_appearance()
        print("✅ Theme switch created successfully")
        
    except Exception as e:
        print(f"⚠️  Error setting up theme switch: {e}")
        import traceback
        traceback.print_exc()

def recreate_theme_switch():
    """Recreate the theme switch after a theme change"""
    global switch_handle, switch_bg
    
    try:
        # Find the existing theme_frame 
        for child in root.winfo_children():
            if hasattr(child, 'grid_info') and child.grid_info().get('row') == 1:
                for subchild in child.winfo_children():
                    if hasattr(subchild, 'winfo_children'):
                        for element in subchild.winfo_children():
                            # Look for our theme frame by checking for moon emoji
                            try:
                                for widget in element.winfo_children():
                                    if hasattr(widget, 'cget') and widget.cget('text') == "🌙":
                                        theme_frame = element
                                        break
                            except:
                                continue
        
        # Clear existing switch components
        for widget in theme_frame.winfo_children():
            if hasattr(widget, 'winfo_children'):
                for child in widget.winfo_children():
                    child.destroy()
                widget.destroy()
        
        # Recreate the switch
        widgets['label'](theme_frame, text="🌙", font=('Segoe UI', 12)).grid(row=0, column=0, padx=(0, 5))

        switch_frame = widgets['frame'](theme_frame)
        switch_frame.grid(row=0, column=1, padx=2)

        # Switch background (acts like the track)
        switch_bg = tk.Frame(switch_frame, width=40, height=20, relief='solid', bd=1)
        switch_bg.grid(row=0, column=0)
        switch_bg.grid_propagate(False)

        # Switch handle (the moving part)
        switch_handle = tk.Button(switch_bg, text="", width=2, height=1, relief='raised', bd=1,
                                 command=lambda: toggle_theme())
        switch_handle.place(x=2, y=2, width=16, height=16)

        widgets['label'](theme_frame, text="☀️", font=('Segoe UI', 12)).grid(row=0, column=2, padx=(5, 0))
        
        # Update appearance to match current theme
        update_theme_switch_appearance()
        
    except Exception as e:
        print(f"⚠️  Error recreating theme switch: {e}")
        # If recreation fails, just update appearance
        update_theme_switch_appearance()

# Theme switch already created above

def update_status():
    """Update the status bar with current information"""
    current_year = get_current_year()
    current_file = data.FILE_PATH.split('\\')[-1] if '\\' in data.FILE_PATH else data.FILE_PATH
    status_var.set(f"Current file: {current_file} | Year: {current_year} | People: {len(data.PEOPLE)}")
    current_year_label.config(text=f"Working on: {current_year}")

# Create notebook for tabs
notebook = widgets['notebook'](main_frame)
notebook.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))

# Initialize Tabelle Manager
tabelle_manager = TabelleManager(main_frame, widgets, colors, theme if THEME_AVAILABLE else None)

# Tab 1: People Management
people_frame = widgets['frame'](notebook, padding="15", card_style=True)
notebook.add(people_frame, text="👥 People Management")

# Left side - Add/Remove people
people_left = widgets['frame'](people_frame, card_style=True)
people_left.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))

widgets['heading_label'](people_left, text="👤 Manage People").grid(row=0, column=0, columnspan=2, pady=(0, 15))

widgets['label'](people_left, text="Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
name_entry = widgets['entry'](people_left, width=20)
name_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))

# Buttons frame with professional spacing - explicitly card style for background consistency
button_frame = widgets['frame'](people_left, card_style=True)
button_frame.grid(row=2, column=0, columnspan=2, pady=20)

add_button = widgets['success_button'](button_frame, text="➕ Add Person", command=lambda: add_person())
add_button.grid(row=0, column=0, padx=(0, 15))

delete_button = widgets['button'](button_frame, text="➖ Remove Person", command=lambda: delete_person())
delete_button.grid(row=0, column=1, padx=(15, 0))

# Experience Level Management Section
exp_frame = widgets['labelframe'](people_left, text="🎯 Experience Level Management", padding="10")
exp_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))

widgets['label'](exp_frame, text="Person:").grid(row=0, column=0, sticky=tk.W, pady=5)
exp_person_var = tk.StringVar()
exp_person_combo = widgets['combobox'](exp_frame, textvariable=exp_person_var, state="readonly", width=18)
exp_person_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))

widgets['label'](exp_frame, text="Experience Level:").grid(row=1, column=0, sticky=tk.W, pady=5)
exp_level_var = tk.StringVar()
exp_level_combo = widgets['combobox'](exp_frame, textvariable=exp_level_var, 
                                      values=["new", "beginner", "learning", "experienced"], 
                                      state="readonly", width=18)
exp_level_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))

# Experience level buttons - frame inside labelframe will auto-detect card style
exp_button_frame = widgets['frame'](exp_frame)
exp_button_frame.grid(row=2, column=0, columnspan=2, pady=10)

set_exp_button = widgets['button'](exp_button_frame, text="🎯 Set Experience Level", command=lambda: set_experience_level())
set_exp_button.grid(row=0, column=0, padx=(0, 15))

remove_exp_button = widgets['button'](exp_button_frame, text="🔄 Reset to Automatic", command=lambda: remove_experience_override())
remove_exp_button.grid(row=0, column=1, padx=(15, 0))

# Configure grid weights for experience frame
exp_frame.columnconfigure(1, weight=1)

# Watering History Balancing Section
balance_frame = widgets['labelframe'](people_left, text="⚖️ Watering History Balancing", padding="10")
balance_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))

# Buttons for balancing - frame inside labelframe will auto-detect card style
balance_button_frame = widgets['frame'](balance_frame)
balance_button_frame.grid(row=0, column=0, columnspan=2, pady=10)

analyze_button = widgets['button'](balance_button_frame, text="📊 Analyze Imbalance", command=lambda: show_watering_analysis())
analyze_button.grid(row=0, column=0, padx=(0, 15))

balance_button = widgets['button'](balance_button_frame, text="⚖️ Balance History", command=lambda: balance_watering_counts())
balance_button.grid(row=0, column=1, padx=(15, 0))

# Configure grid weights for balance frame
balance_frame.columnconfigure(0, weight=1)
balance_frame.columnconfigure(1, weight=1)

# Right side - People list with details
people_right = widgets['frame'](people_frame, card_style=True)
people_right.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

widgets['heading_label'](people_right, text="📊 Current People & Statistics").grid(row=0, column=0, pady=(0, 15))

# Create treeview for people list
people_tree = widgets['treeview'](people_right, columns=('Name', 'Watering Count', 'Experience Level', 'Weight', 'Extra Weight'), show='headings', height=12)
people_tree.heading('Name', text='Name')
people_tree.heading('Watering Count', text='Times Watered')
people_tree.heading('Experience Level', text='Experience Level')
people_tree.heading('Weight', text='Weight')
people_tree.heading('Extra Weight', text='Extra Weight')
people_tree.column('Name', width=120)
people_tree.column('Watering Count', width=80)
people_tree.column('Experience Level', width=100)
people_tree.column('Weight', width=60)
people_tree.column('Extra Weight', width=70)
people_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Scrollbar for treeview
people_scrollbar = widgets['scrollbar'](people_right, orient=tk.VERTICAL, command=people_tree.yview)
people_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
people_tree.configure(yscrollcommand=people_scrollbar.set)

# Configure grid weights for people frame
people_frame.columnconfigure(0, weight=1)
people_frame.columnconfigure(1, weight=2)
people_frame.rowconfigure(0, weight=1)
people_left.columnconfigure(1, weight=1)
people_right.columnconfigure(0, weight=1)
people_right.rowconfigure(1, weight=1)

def add_person():
    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name.")
        return
    
    # Allow German umlauts in name validation
    normalized_for_validation = data.normalize_german_name(name)
    if not normalized_for_validation.replace(" ", "").isalpha():
        messagebox.showerror("Error", "Name should contain only letters (German umlauts are automatically converted).")
        return
    
    # Check if name will be normalized (contains German umlauts)
    normalized_name = data.normalize_german_name(name)
    name_changed = name != normalized_name
    
    if normalized_name in data.PEOPLE:
        messagebox.showerror("Error", "Person already exists.")
        return
        
    if add_new_person_with_context(name):
        update_people_list()
        refresh_dependencies()
        name_entry.delete(0, tk.END)
        update_status()
        
        # Show different message if name was normalized
        if name_changed:
            messagebox.showinfo("Success", f"Added '{normalized_name}' (German umlauts converted) with context-appropriate weight.")
        else:
            messagebox.showinfo("Success", f"Added {normalized_name} with context-appropriate weight.")
    else:
        messagebox.showerror("Error", "Failed to add person.")

def delete_person():
    name = name_entry.get().strip()
    if not name:
        # Try to get selection from treeview
        selection = people_tree.selection()
        if selection:
            name = people_tree.item(selection[0])['values'][0]
        else:
            messagebox.showerror("Error", "Please enter a name or select from the list.")
            return
    
    if name not in data.PEOPLE:
        messagebox.showerror("Error", "Person not found.")
        return
        
    # Confirmation dialog
    if messagebox.askyesno("Confirm Delete", f"Are you sure you want to remove {name}?"):
        if remove_person_and_rebalance(name):
            update_people_list()
            refresh_dependencies()
            name_entry.delete(0, tk.END)
            update_status()
            messagebox.showinfo("Success", f"Removed {name} and rebalanced system.")
        else:
            messagebox.showerror("Error", "Failed to remove person.")

def set_experience_level():
    """Set manual experience level for a person"""
    person = exp_person_var.get().strip()
    level = exp_level_var.get().strip()
    
    if not person:
        messagebox.showerror("Error", "Please select a person.")
        return
    
    if not level:
        messagebox.showerror("Error", "Please select an experience level.")
        return
    
    if set_person_experience_level(person, level):
        update_people_list()
        update_status()
        messagebox.showinfo("Success", f"Set {person}'s experience level to '{level}'.")
    else:
        messagebox.showerror("Error", "Failed to set experience level.")

def remove_experience_override():
    """Remove manual experience level override for a person"""
    person = exp_person_var.get().strip()
    
    if not person:
        messagebox.showerror("Error", "Please select a person.")
        return
    
    if remove_person_experience_override(person):
        update_people_list()
        update_status()
        messagebox.showinfo("Success", f"Reset {person}'s experience level to automatic calculation.")
    else:
        messagebox.showinfo("Info", f"No manual override found for {person}.")

def show_watering_analysis():
    """Show detailed analysis of watering history imbalance"""
    try:
        report = get_watering_history_report()
        
        # Create a new window to display the analysis
        analysis_window = tk.Toplevel(root)
        analysis_window.title("Watering History Analysis")
        analysis_window.geometry("600x500")
        
        # Create text widget with scrollbar
        text_frame = widgets['frame'](analysis_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        theme_instance.configure_text_widget(text_widget)
        scrollbar = widgets['scrollbar'](text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert the report
        text_widget.insert(tk.END, report)
        text_widget.configure(state=tk.DISABLED)
        
        # Add buttons
        button_frame = widgets['frame'](analysis_window)
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        balance_btn = widgets['button'](button_frame, text="⚖️ Balance Now", 
                                               command=lambda: [balance_watering_counts(), analysis_window.destroy()])
        balance_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        close_btn = widgets['button'](button_frame, text="Close", command=analysis_window.destroy)
        close_btn.pack(side=tk.RIGHT)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate analysis: {str(e)}")

def balance_watering_counts():
    """Balance the watering counts across all people"""
    try:
        # First check if balancing is needed
        analysis = analyze_watering_imbalance()
        if not analysis:
            messagebox.showerror("Error", "No watering data available to balance.")
            return
        
        if analysis['difference'] <= 2:
            messagebox.showinfo("Info", 
                              f"Watering counts are already balanced.\n"
                              f"Current range: {analysis['min_count']} - {analysis['max_count']} (difference: {analysis['difference']})")
            return
        
        # Show confirmation dialog with current imbalance
        confirm_msg = (f"Current watering distribution:\n"
                      f"Range: {analysis['min_count']} - {analysis['max_count']} (difference: {analysis['difference']})\n"
                      f"Average: {analysis['average']:.1f}\n\n"
                      f"This will redistribute watering entries to balance the counts.\n"
                      f"Do you want to proceed?")
        
        if not messagebox.askyesno("Confirm Balancing", confirm_msg):
            return
        
        # Perform the balancing
        success, message = balance_watering_history()
        
        if success:
            # Update all displays
            update_people_list()
            update_schedule_display()
            update_status()
            messagebox.showinfo("Success", f"Watering history balanced successfully!\n\n{message}")
        else:
            messagebox.showinfo("Info", message)
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to balance watering history: {str(e)}")

def update_people_list():
    # Clear existing items
    for item in people_tree.get_children():
        people_tree.delete(item)
    
    # Ensure watering history is up to date
    for person in data.PEOPLE:
        if person not in data.watering_history:
            data.watering_history[person] = []
    for person in list(data.watering_history.keys()):
        if person not in data.PEOPLE:
            del data.watering_history[person]
    
    # Update weights to ensure they reflect current watering counts
    data.update_weights()
    
    # Add people to treeview
    for i, person in enumerate(data.PEOPLE):
        watering_count = len(data.watering_history.get(person, []))
        experience_level = get_person_experience_level(person)
        # Add indicator for manual override
        if person in data.experience_overrides:
            experience_level += " (Manual)"
        weight = data.WEIGHTS[i] if i < len(data.WEIGHTS) else 1
        extra_weight = data.EXTRA_WEIGHTS[i] if i < len(data.EXTRA_WEIGHTS) else 1
        people_tree.insert('', 'end', values=(person, watering_count, experience_level, weight, extra_weight))
    
    # Update person combos when people list changes
    update_person_combos()

# Tab 2: Schedule Generation  
schedule_frame = widgets['frame'](notebook, padding="15", card_style=True)
notebook.add(schedule_frame, text="📅 Schedule Generation")# Schedule generation section
schedule_gen_frame = widgets['labelframe'](schedule_frame, text="🔄 Generate New Schedule", padding="15")
schedule_gen_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))

# Schedule controls frame
schedule_controls = widgets['frame'](schedule_gen_frame)
schedule_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

widgets['label'](schedule_controls, text="Schedule Options:").grid(row=0, column=0, padx=(0, 10))

# Dropdown for schedule type
schedule_type_var = tk.StringVar(value="Next 6 Weeks")
schedule_type_combo = widgets['combobox'](schedule_controls, textvariable=schedule_type_var, 
                                  values=["Next 6 Weeks", "Remaining Weeks"], 
                                  state="readonly", width=15)
schedule_type_combo.grid(row=0, column=1, padx=(0, 15))

generate_button = widgets['button'](schedule_controls, text="🔄 Generate Schedule", command=lambda: generate_and_show_schedule())
generate_button.grid(row=0, column=2, padx=(15, 0))

refresh_button = widgets['button'](schedule_controls, text="🔄 Refresh Display", command=lambda: update_all_displays())
refresh_button.grid(row=0, column=3, padx=(10, 0))

# Schedule display - create container for both sections
schedule_display_container = widgets['frame'](schedule_frame)
schedule_display_container.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))

# Current Schedule section
current_schedule_frame = widgets['labelframe'](schedule_display_container, text="📋 Current Schedule", padding="15")
current_schedule_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

# Create a frame to hold the treeview and scrollbar
current_schedule_content = widgets['frame'](current_schedule_frame)
current_schedule_content.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Treeview for schedule display
schedule_tree = widgets['treeview'](current_schedule_content, columns=('Week', 'Date Range', 'Person 1', 'Person 2', 'ErsatzPerson 1', 'ErsatzPerson 2'), show='headings', height=10)
schedule_tree.heading('Week', text='Week')
schedule_tree.heading('Date Range', text='Date Range')
schedule_tree.heading('Person 1', text='Person 1')
schedule_tree.heading('Person 2', text='Person 2')
schedule_tree.heading('ErsatzPerson 1', text='ErsatzPerson 1')
schedule_tree.heading('ErsatzPerson 2', text='ErsatzPerson 2')
schedule_tree.column('Week', width=60)
schedule_tree.column('Date Range', width=120)
schedule_tree.column('Person 1', width=100)
schedule_tree.column('Person 2', width=100)
schedule_tree.column('ErsatzPerson 1', width=100)
schedule_tree.column('ErsatzPerson 2', width=100)
schedule_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

# Configure alternating row colors with theme
theme_instance.configure_treeview_tags(schedule_tree)

# Scrollbar for schedule treeview
schedule_tree_scrollbar = widgets['scrollbar'](current_schedule_content, orient=tk.VERTICAL, command=schedule_tree.yview)
schedule_tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
schedule_tree.configure(yscrollcommand=schedule_tree_scrollbar.set)

# Schedule Summary section - now at the same level as Current Schedule
schedule_summary_frame = widgets['labelframe'](schedule_display_container, text="📊 Schedule Summary", padding="15")
schedule_summary_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

# Create a frame for the canvas and scrollbar inside the labelframe
summary_content_frame = widgets['frame'](schedule_summary_frame)
summary_content_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create a canvas for visual schedule representation - let it expand to match the treeview height
schedule_canvas = tk.Canvas(summary_content_frame, width=300)
theme_instance.configure_canvas(schedule_canvas)
schedule_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Canvas scrollbar
canvas_scrollbar = widgets['scrollbar'](summary_content_frame, orient=tk.VERTICAL, command=schedule_canvas.yview)
canvas_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
schedule_canvas.configure(yscrollcommand=canvas_scrollbar.set)

# Configure grid weights for schedule display
schedule_display_container.columnconfigure(0, weight=2)  # Current Schedule gets more space
schedule_display_container.columnconfigure(1, weight=1)  # Schedule Summary gets less space
schedule_display_container.rowconfigure(0, weight=1)
current_schedule_frame.columnconfigure(0, weight=1)
current_schedule_frame.rowconfigure(0, weight=1)
current_schedule_content.columnconfigure(0, weight=1)
current_schedule_content.rowconfigure(0, weight=1)
schedule_summary_frame.columnconfigure(0, weight=1)
schedule_summary_frame.rowconfigure(0, weight=1)
summary_content_frame.columnconfigure(0, weight=1)
summary_content_frame.rowconfigure(0, weight=1)

# Configure grid weights for schedule frame
schedule_frame.columnconfigure(0, weight=1)
schedule_frame.rowconfigure(1, weight=1)

def update_schedule_display():
    """Update the schedule display with current data"""
    # Clear existing items
    for item in schedule_tree.get_children():
        schedule_tree.delete(item)
    
    # Clear canvas
    schedule_canvas.delete("all")
    
    # Get current year from the selected data
    current_year = datetime.date.today().year
    if hasattr(data, 'FILE_PATH') and 'people_' in data.FILE_PATH:
        import re
        match = re.search(r'people_(\d{4})\.json', data.FILE_PATH)
        if match:
            current_year = int(match.group(1))
    
    # Collect all schedule entries from watering history
    week_assignments = {}  # week_number: {'main': [person1, person2], 'ersatz': [ersatz1, ersatz2]}
    
    for person in data.PEOPLE:
        if person in data.watering_history:
            for entry in data.watering_history[person]:
                if "KW" in entry:
                    try:
                        # Parse new KW format: "2025 KW 15: person1 and person2 (ErsatzPersons: ersatz1 and ersatz2)"
                        parts = entry.split(":")
                        if len(parts) >= 2:
                            week_part = parts[0].strip()
                            # Extract year and week number
                            if "KW" in week_part:
                                import re
                                match = re.search(r'(\d{4})\s*KW\s*(\d+)', week_part)
                                if match:
                                    year_num = int(match.group(1))
                                    week_num = int(match.group(2))
                                    
                                    # Only show entries for the current year being viewed
                                    if year_num == current_year:
                                        if week_num not in week_assignments:
                                            week_assignments[week_num] = {'main': [], 'ersatz': []}
                                        
                                        # Parse main persons and ersatz persons
                                        if person not in week_assignments[week_num]['main']:
                                            week_assignments[week_num]['main'].append(person)
                                        
                                        # Try to extract ErsatzPersons from the entry
                                        if "ErsatzPersons:" in entry:
                                            ersatz_part = entry.split("ErsatzPersons:")[1].strip()
                                            if " and " in ersatz_part:
                                                ersatz_persons = ersatz_part.rstrip(")").split(" and ")
                                                if len(ersatz_persons) >= 2:
                                                    ersatz1 = ersatz_persons[0].strip()
                                                    ersatz2 = ersatz_persons[1].strip()
                                                    if ersatz1 not in week_assignments[week_num]['ersatz']:
                                                        week_assignments[week_num]['ersatz'].append(ersatz1)
                                                    if ersatz2 not in week_assignments[week_num]['ersatz']:
                                                        week_assignments[week_num]['ersatz'].append(ersatz2)
                    except (ValueError, IndexError):
                        continue
                elif entry.startswith("Week "):  # Support old format for backward compatibility
                    try:
                        week_part = entry.split(":")[0]
                        week_num = int(week_part.split()[1])
                        
                        if week_num not in week_assignments:
                            week_assignments[week_num] = {'main': [], 'ersatz': []}
                        
                        if person not in week_assignments[week_num]['main']:
                            week_assignments[week_num]['main'].append(person)
                    except (ValueError, IndexError):
                        continue
    
    # Sort weeks and create display entries
    sorted_weeks = sorted(week_assignments.keys())
    current_week = datetime.date.today().isocalendar()[1]
    actual_current_year = datetime.date.today().year
    
    for i, week_num in enumerate(sorted_weeks):
        # Calculate date range for the week
        try:
            jan_1 = datetime.date(current_year, 1, 1)
            week_start = jan_1 + datetime.timedelta(weeks=week_num-1)
            week_start = week_start - datetime.timedelta(days=week_start.weekday())
            week_end = week_start + datetime.timedelta(days=6)
            date_range = f"{week_start.strftime('%m/%d')} - {week_end.strftime('%m/%d')}"
        except:
            date_range = "TBD"
        
        # Get people assigned to this week
        assignment = week_assignments[week_num]
        main_people = assignment.get('main', [])
        ersatz_people = assignment.get('ersatz', [])
        
        person1 = main_people[0] if len(main_people) > 0 else ""
        person2 = main_people[1] if len(main_people) > 1 else ""
        ersatz_person1 = ersatz_people[0] if len(ersatz_people) > 0 else ""
        ersatz_person2 = ersatz_people[1] if len(ersatz_people) > 1 else ""
        
        # Determine row styling - only highlight current/next week if viewing the actual current year
        if current_year == actual_current_year:
            if week_num == current_week:
                tag = 'current_week'
            elif week_num == current_week + 1:
                tag = 'next_week'
            else:
                tag = 'oddrow' if i % 2 == 0 else 'evenrow'
        else:
            # For past or future years, just use alternating row colors
            tag = 'oddrow' if i % 2 == 0 else 'evenrow'
        
        # Insert into treeview
        schedule_tree.insert('', 'end', values=(f"KW {week_num}", date_range, person1, person2, ersatz_person1, ersatz_person2), tags=(tag,))
    
    # Draw canvas visualization
    draw_schedule_visualization(sorted_weeks, week_assignments, current_week, current_year, actual_current_year)

def draw_schedule_visualization(sorted_weeks, week_assignments, current_week, current_year, actual_current_year):
    """Draw a visual representation of the schedule on the canvas"""
    canvas_width = 280
    # Calculate content height based on number of weeks for scrolling
    canvas_height = len(sorted_weeks) * 60 + 40
    
    # Update canvas scroll region
    schedule_canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
    
    # Get theme colors if available
    if THEME_AVAILABLE:
        canvas_colors = theme.get_canvas_colors()
        person_colors = {person: theme.get_person_color(i) for i, person in enumerate(data.PEOPLE)}
    else:
        canvas_colors = {
            'background': '#ffffff',
            'border': '#cccccc',
            'current_week': '#e6f3ff',
            'current_week_border': '#0066cc',
            'next_week': '#ffe6cc',
            'next_week_border': '#ff8800',
            'text': '#333333',
            'title': '#0066cc',
            'legend': '#666666'
        }
        colors_list = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0', '#ffb3e6', '#c4e17f']
        person_colors = {person: colors_list[i % len(colors_list)] for i, person in enumerate(data.PEOPLE)}
    
    # Draw week blocks (removed "Visual Schedule" title)
    y_offset = 20  # Reduced offset since no title
    block_height = 50
    block_width = 260
    
    for i, week_num in enumerate(sorted_weeks):
        # Determine block color based on week status - only highlight current/next week if viewing the actual current year
        if current_year == actual_current_year:
            if week_num == current_week:
                border_color = canvas_colors['current_week_border']
                border_width = 3
                bg_color = canvas_colors['current_week']
            elif week_num == current_week + 1:
                border_color = canvas_colors['next_week_border']
                border_width = 2
                bg_color = canvas_colors['next_week']
            else:
                border_color = canvas_colors['border']
                border_width = 1
                bg_color = canvas_colors['background']
        else:
            # For past or future years, just use normal styling
            border_color = canvas_colors['border']
            border_width = 1
            bg_color = canvas_colors['background']
        
        # Draw main block
        schedule_canvas.create_rectangle(10, y_offset, 10 + block_width, y_offset + block_height, 
                                       outline=border_color, width=border_width, fill=bg_color)
        
        # Draw week number
        schedule_canvas.create_text(30, y_offset + 15, text=f"Week {week_num}", 
                                  font=theme_instance.get_canvas_font('heading'), anchor='w', fill=canvas_colors['text'])
        
        # Get people for this week
        assignment = week_assignments.get(week_num, {'main': [], 'ersatz': []})
        main_people = assignment.get('main', [])
        
        if len(main_people) >= 2:
            person1, person2 = main_people[0], main_people[1]
            
            # Draw person 1 section
            person1_color = person_colors.get(person1, colors.LIGHT_GRAY)
            schedule_canvas.create_rectangle(15, y_offset + 25, 135, y_offset + 45, 
                                           fill=person1_color, outline=colors.DARK_GRAY, width=1)
            schedule_canvas.create_text(75, y_offset + 35, text=person1, 
                                      font=theme_instance.get_canvas_font('text'), anchor='center', fill=canvas_colors['text'])
            
            # Draw person 2 section
            person2_color = person_colors.get(person2, colors.LIGHT_GRAY)
            schedule_canvas.create_rectangle(140, y_offset + 25, 260, y_offset + 45, 
                                           fill=person2_color, outline=colors.DARK_GRAY, width=1)
            schedule_canvas.create_text(200, y_offset + 35, text=person2, 
                                      font=theme_instance.get_canvas_font('text'), anchor='center', fill=canvas_colors['text'])
        
        y_offset += block_height + 10
    
    # Draw legend
    legend_y = y_offset + 20
    schedule_canvas.create_text(15, legend_y, text="Legend:", 
                              font=theme_instance.get_canvas_font('heading'), anchor='w', fill=canvas_colors['legend'])
    
    # Current week indicator
    schedule_canvas.create_rectangle(15, legend_y + 15, 25, legend_y + 25, 
                                   outline=canvas_colors['current_week_border'], width=3, 
                                   fill=canvas_colors['current_week'])
    schedule_canvas.create_text(30, legend_y + 20, text="Current Week", 
                              font=theme_instance.get_canvas_font('legend'), anchor='w', fill=canvas_colors['text'])
    
    # Next week indicator
    schedule_canvas.create_rectangle(15, legend_y + 35, 25, legend_y + 45, 
                                   outline=canvas_colors['next_week_border'], width=2, 
                                   fill=canvas_colors['next_week'])
    schedule_canvas.create_text(30, legend_y + 40, text="Next Week", 
                              font=theme_instance.get_canvas_font('legend'), anchor='w', fill=canvas_colors['text'])

def generate_and_show_schedule():
    try:
        schedule_type = schedule_type_var.get()
        
        # Import the schedule generation function
        from schedule import generate_schedule
        
        # Generate schedule with the selected type
        new_schedule = generate_schedule(schedule_type)
        
        if new_schedule:
            # Extract week information from the generated schedule
            week_info = []
            current_year = None
            week_numbers = []
            
            for entry in new_schedule:
                # Extract year and week number from entries like "2025 KW 48: Jan and Rosa"
                import re
                match = re.search(r'(\d{4}) KW (\d+)', entry)
                if match:
                    year = int(match.group(1))
                    week_num = int(match.group(2))
                    
                    if current_year is None:
                        current_year = year
                        week_numbers = [week_num]
                    elif year == current_year:
                        week_numbers.append(week_num)
                    else:
                        # Year changed, add previous year's info
                        if week_numbers:
                            if len(week_numbers) == 1:
                                week_info.append(f"{current_year}: KW {week_numbers[0]}")
                            else:
                                week_info.append(f"{current_year}: KW {min(week_numbers)}-{max(week_numbers)}")
                        
                        # Start new year
                        current_year = year
                        week_numbers = [week_num]
            
            # Add the last year's info
            if current_year and week_numbers:
                if len(week_numbers) == 1:
                    week_info.append(f"{current_year}: KW {week_numbers[0]}")
                else:
                    week_info.append(f"{current_year}: KW {min(week_numbers)}-{max(week_numbers)}")
            
            # Create success message with week ranges
            success_msg = f"Generate {schedule_type} schedule with {len(new_schedule)} weeks"
            if week_info:
                success_msg += f"\n\nWeeks generated:\n• " + "\n• ".join(week_info)
            
            # Show success message
            messagebox.showinfo("Success", success_msg)
        else:
            messagebox.showinfo("Info", "Schedule generation completed")
        
        # Update all displays - including year selection in case new year files were created
        refresh_years()
        update_schedule_display()
        update_people_list()
        update_status()
        # Update tabelle management displays
        if 'tabelle_manager' in globals():
            tabelle_manager.update_displays()
        
    except PermissionError:
        messagebox.showerror("File Permission Error", 
                           "Cannot access the file - it may be open in another application.\n\n"
                           "Please close any Excel files or other applications using this file and try again.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate schedule: {str(e)}")

# Tab 3: Manual Schedule Management
manual_frame = widgets['frame'](notebook, padding="15", card_style=True)
notebook.add(manual_frame, text="✏️ Manual Management")

# Tab 4: Tabelle Management
tabelle_manager.create_tabelle_tab(notebook)

# Manual date/week management
manual_mgmt_frame = widgets['labelframe'](manual_frame, text="✏️ Add/Remove Specific Dates or Weeks", padding="15")
manual_mgmt_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

# Calendar week and year selection
widgets['label'](manual_mgmt_frame, text="Calendar Week:").grid(row=0, column=0, sticky=tk.W, pady=8)
week_var = tk.StringVar()
week_combo = widgets['combobox'](manual_mgmt_frame, textvariable=week_var, width=10, state="readonly")
week_combo['values'] = [f"KW {i}" for i in range(1, 53)]
week_combo.grid(row=0, column=1, sticky=tk.W, pady=8, padx=(10, 0))

widgets['label'](manual_mgmt_frame, text="Year:").grid(row=0, column=2, sticky=tk.W, pady=8, padx=(20, 0))
manual_year_var = tk.StringVar()
manual_year_combo = widgets['combobox'](manual_mgmt_frame, textvariable=manual_year_var, width=10, state="readonly")
manual_year_combo.grid(row=0, column=3, sticky=tk.W, pady=8, padx=(10, 0))

widgets['label'](manual_mgmt_frame, text="Person 1:").grid(row=1, column=0, sticky=tk.W, pady=8)
person1_var = tk.StringVar()
person1_combo = widgets['combobox'](manual_mgmt_frame, textvariable=person1_var, width=18, state="readonly")
person1_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=8, padx=(10, 0))

widgets['label'](manual_mgmt_frame, text="Person 2:").grid(row=1, column=2, sticky=tk.W, pady=8, padx=(20, 0))
person2_var = tk.StringVar()
person2_combo = widgets['combobox'](manual_mgmt_frame, textvariable=person2_var, width=18, state="readonly")
person2_combo.grid(row=1, column=3, sticky=(tk.W, tk.E), pady=8, padx=(10, 0))

# ErsatzPerson (substitute person) fields
widgets['label'](manual_mgmt_frame, text="ErsatzPerson 1:").grid(row=2, column=0, sticky=tk.W, pady=8)
ersatz_person1_var = tk.StringVar()
ersatz_person1_combo = widgets['combobox'](manual_mgmt_frame, textvariable=ersatz_person1_var, width=18, state="readonly")
ersatz_person1_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=8, padx=(10, 0))

widgets['label'](manual_mgmt_frame, text="ErsatzPerson 2:").grid(row=2, column=2, sticky=tk.W, pady=8, padx=(20, 0))
ersatz_person2_var = tk.StringVar()
ersatz_person2_combo = widgets['combobox'](manual_mgmt_frame, textvariable=ersatz_person2_var, width=18, state="readonly")
ersatz_person2_combo.grid(row=2, column=3, sticky=(tk.W, tk.E), pady=8, padx=(10, 0))

manual_button_frame = widgets['frame'](manual_mgmt_frame)
manual_button_frame.grid(row=3, column=0, columnspan=4, pady=15)

add_date_button = widgets['success_button'](manual_button_frame, text="➕ Add Date/Week", command=lambda: add_date_or_week())
add_date_button.grid(row=0, column=0, padx=(0, 15))

delete_date_button = widgets['button'](manual_button_frame, text="➖ Delete Date/Week", command=lambda: delete_date_or_week())
delete_date_button.grid(row=0, column=1, padx=(15, 0))

# Configure grid weights for manual frame
manual_mgmt_frame.columnconfigure(1, weight=1)
manual_mgmt_frame.columnconfigure(3, weight=1)

# Data Backup and Recovery System
backup_frame = widgets['labelframe'](manual_frame, text="💾 Data Backup & Recovery", padding="15")
backup_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

# Backup section
backup_info_frame = widgets['frame'](backup_frame)
backup_info_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

backup_info_label = widgets['label'](backup_info_frame, 
    text="Create and manage data backups for recovery after code reinstallation.")
backup_info_label.grid(row=0, column=0, sticky=tk.W)

# Template status
template_status_frame = widgets['frame'](backup_frame)
template_status_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 10))

template_status_label = widgets['label'](template_status_frame, text="Template Status:")
template_status_label.grid(row=0, column=0, sticky=tk.W)

template_status_value = widgets['label'](template_status_frame, text="Checking...")
template_status_value.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

# Backup buttons
backup_button_frame = widgets['frame'](backup_frame)
backup_button_frame.grid(row=2, column=0, columnspan=3, pady=10)

def backup_current_data():
    """Backup current data to people.json template"""
    if BACKUP_AVAILABLE:
        try:
            if data_backup_recovery.backup_current_year_to_template():
                update_template_status()
                messagebox.showinfo("Backup Complete", 
                    "✅ Successfully backed up current data to people.json template!\n\n"
                    "This template can now be used for data recovery.")
            else:
                messagebox.showwarning("Backup Cancelled", "Backup operation was cancelled.")
        except Exception as e:
            messagebox.showerror("Backup Error", f"Failed to backup data: {str(e)}")
    else:
        messagebox.showerror("Not Available", "Backup system not available.")

def restore_from_backup():
    """Restore data from people.json template"""
    if BACKUP_AVAILABLE:
        try:
            current_year = data.get_current_year()
            restore_msg = f"""
Restore data from people.json template to {current_year}?

This will:
• Replace current {current_year} people data
• Restore balanced weights from template
• Restore experience levels
• Clear current watering history

Current data will be overwritten!
"""
            if messagebox.askyesno("Confirm Restore", restore_msg):
                if data_backup_recovery.restore_from_template(current_year):
                    # Refresh all displays
                    update_all_displays()
                    messagebox.showinfo("Restore Complete", 
                        f"✅ Successfully restored data from template to {current_year}!\n\n"
                        f"Data has been refreshed.")
                else:
                    messagebox.showerror("Restore Failed", "Failed to restore from template.")
        except Exception as e:
            messagebox.showerror("Restore Error", f"Failed to restore data: {str(e)}")
    else:
        messagebox.showerror("Not Available", "Backup system not available.")

def check_data_integrity():
    """Check data integrity and suggest fixes"""
    if BACKUP_AVAILABLE:
        try:
            # Capture the integrity check output
            import io
            import sys
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                integrity_ok = data_backup_recovery.check_data_integrity()
            
            output = f.getvalue()
            
            if integrity_ok:
                messagebox.showinfo("Data Integrity Check", 
                    "✅ Data integrity check passed!\n\nNo issues found with your data.")
            else:
                messagebox.showwarning("Data Integrity Issues", 
                    f"⚠️  Data integrity issues found:\n\n{output}\n\n"
                    f"Consider running the suggested fixes or restoring from backup.")
        except Exception as e:
            messagebox.showerror("Integrity Check Error", f"Failed to check data integrity: {str(e)}")
    else:
        messagebox.showerror("Not Available", "Backup system not available.")

def create_migration_package():
    """Create a complete migration package"""
    if BACKUP_AVAILABLE:
        try:
            package_path = data_backup_recovery.migrate_data_package()
            if package_path:
                messagebox.showinfo("Migration Package Created", 
                    f"✅ Migration package created successfully!\n\n"
                    f"Location: {package_path}\n\n"
                    f"This package contains all your data files and can be used to "
                    f"migrate to a new installation.")
            else:
                messagebox.showerror("Migration Failed", "Failed to create migration package.")
        except Exception as e:
            messagebox.showerror("Migration Error", f"Failed to create migration package: {str(e)}")
    else:
        messagebox.showerror("Not Available", "Backup system not available.")

backup_button = widgets['success_button'](backup_button_frame, text="💾 Backup Current Data", 
                                          command=backup_current_data)
backup_button.grid(row=0, column=0, padx=(0, 10))

restore_button = widgets['button'](backup_button_frame, text="🔄 Restore from Backup", 
                                  command=restore_from_backup)
restore_button.grid(row=0, column=1, padx=(10, 10))

integrity_button = widgets['button'](backup_button_frame, text="🔍 Check Integrity", 
                                    command=check_data_integrity)
integrity_button.grid(row=0, column=2, padx=(10, 10))

migrate_button = widgets['button'](backup_button_frame, text="📦 Create Migration Package", 
                                  command=create_migration_package)
migrate_button.grid(row=0, column=3, padx=(10, 0))

def update_template_status():
    """Update the template status display"""
    import os
    if os.path.exists("people.json"):
        try:
            import json
            with open("people.json", "r", encoding='utf-8') as f:
                template = json.load(f)
            
            people_count = len(template.get("PEOPLE", []))
            metadata = template.get("METADATA", {})
            source_year = metadata.get("source_year", "unknown")
            
            status_text = f"✅ Available ({people_count} people from {source_year})"
            template_status_value.config(text=status_text, foreground="green")
        except:
            template_status_value.config(text="⚠️ Corrupted", foreground="orange")
    else:
        template_status_value.config(text="❌ Not Found", foreground="red")

# Configure grid weights for backup frame
backup_frame.columnconfigure(0, weight=1)

def update_person_combos():
    """Update the combobox options with current people"""
    print(f"Updating combos with people: {data.PEOPLE}")
    person1_combo['values'] = data.PEOPLE
    person2_combo['values'] = data.PEOPLE
    ersatz_person1_combo['values'] = data.PEOPLE
    ersatz_person2_combo['values'] = data.PEOPLE
    
    # Update experience level management combo
    exp_person_combo['values'] = data.PEOPLE
    if data.PEOPLE and not exp_person_var.get():
        exp_person_var.set(data.PEOPLE[0])
    
    # Clear any existing selections that might be invalid
    if person1_var.get() not in data.PEOPLE:
        person1_var.set("")
    if person2_var.get() not in data.PEOPLE:
        person2_var.set("")
    if ersatz_person1_var.get() not in data.PEOPLE:
        ersatz_person1_var.set("")
    if ersatz_person2_var.get() not in data.PEOPLE:
        ersatz_person2_var.set("")
    
    # Update manual year combo with available years
    available_years = data.get_available_years()
    manual_year_combo['values'] = available_years
    if available_years:
        manual_year_var.set(available_years[-1])  # Set to latest year

def add_date_or_week():
    week_selection = week_var.get().strip()
    year_selection = manual_year_var.get().strip()

    print(f"Selected Week: {week_selection}, Selected Year: {year_selection}")

    person1 = person1_var.get().strip()
    person2 = person2_var.get().strip()
    ersatz_person1 = ersatz_person1_var.get().strip()
    ersatz_person2 = ersatz_person2_var.get().strip()

    # Clean up person names in case they contain extra text
    # Remove any parentheses and content within them
    import re
    person1 = re.sub(r'\s*\([^)]*\).*', '', person1).strip()
    person2 = re.sub(r'\s*\([^)]*\).*', '', person2).strip()
    ersatz_person1 = re.sub(r'\s*\([^)]*\).*', '', ersatz_person1).strip()
    ersatz_person2 = re.sub(r'\s*\([^)]*\).*', '', ersatz_person2).strip()

    print(f"Cleaned names - Person1: '{person1}', Person2: '{person2}', Ersatz1: '{ersatz_person1}', Ersatz2: '{ersatz_person2}'")
    print(f"Available people: {data.PEOPLE}")

    if not week_selection or not year_selection or not person1 or not person2:
        messagebox.showerror("Error", "Please fill in all main person fields.")
        return

    if person1 == person2:
        messagebox.showerror("Error", "Please select two different main people.")
        return

    if person1 not in data.PEOPLE or person2 not in data.PEOPLE:
        messagebox.showerror("Error", "Please select valid main people from the list.")
        return

    # Check ersatz persons if they are filled
    if ersatz_person1 and ersatz_person1 not in data.PEOPLE:
        messagebox.showerror("Error", "Please select valid ErsatzPerson 1 from the list.")
        return
    
    if ersatz_person2 and ersatz_person2 not in data.PEOPLE:
        messagebox.showerror("Error", "Please select valid ErsatzPerson 2 from the list.")
        return

    if ersatz_person1 and ersatz_person2 and ersatz_person1 == ersatz_person2:
        messagebox.showerror("Error", "Please select two different ErsatzPersons.")
        return

    # Ensure ErsatzPersons are not the same as main persons
    if ersatz_person1 and (ersatz_person1 == person1 or ersatz_person1 == person2):
        messagebox.showerror("Error", "ErsatzPerson 1 cannot be the same as a main person.")
        return
        
    if ersatz_person2 and (ersatz_person2 == person1 or ersatz_person2 == person2):
        messagebox.showerror("Error", "ErsatzPerson 2 cannot be the same as a main person.")
        return

    # Extract week number from KW format (e.g., "KW 15" -> "15")
    week_number = week_selection.replace("KW ", "").strip()

    # Check if week already has existing data
    existing_data = get_week_data_with_ersatz(year_selection, week_number)
    print(f"Existing Data for Week {week_number}: {existing_data}")
    
    # If there's existing data, show confirmation dialog
    if existing_data[0] or existing_data[1]:
        existing_person1 = existing_data[0] or "Unknown"
        existing_person2 = existing_data[1] or "Unknown"
        
        confirmation_message = f"Are you sure you want to change {week_selection} {year_selection} from {existing_person1} and {existing_person2} to {person1} and {person2}?"
        
        if not messagebox.askyesno("Confirm Change", confirmation_message):
            return

    # Update week data with ErsatzPersons
    update_week_data_with_ersatz(year_selection, week_number, person1, person2, ersatz_person1, ersatz_person2)

    # Reload the current data to reflect the changes in the global variables
    reload_current_data()

    # Excel functionality removed - using JSON-only data storage

    # Clear entries
    week_var.set("")
    manual_year_var.set("")
    person1_var.set("")
    person2_var.set("")
    ersatz_person1_var.set("")
    ersatz_person2_var.set("")

    update_people_list()
    update_schedule_display()
    update_status()
    # Update tabelle management displays
    if 'tabelle_manager' in globals():
        tabelle_manager.update_displays()
    
    # Create success message
    success_message = f"Entry for {year_selection} {week_selection} added successfully."
    if ersatz_person1 or ersatz_person2:
        ersatz_info = []
        if ersatz_person1:
            ersatz_info.append(f"ErsatzPerson 1: {ersatz_person1}")
        if ersatz_person2:
            ersatz_info.append(f"ErsatzPerson 2: {ersatz_person2}")
        success_message += f"\nErsatzPersons: {', '.join(ersatz_info)}"
    
    messagebox.showinfo("Success", success_message)

def delete_date_or_week():
    week_selection = week_var.get().strip()
    year_selection = manual_year_var.get().strip()

    if not week_selection or not year_selection:
        messagebox.showerror("Error", "Please select both calendar week and year to delete.")
        return
    
    # Extract week number from KW format
    week_number = week_selection.replace("KW ", "").strip()
    
    # Create search pattern for deletion
    search_pattern = f"{year_selection} {week_selection}:"
        
    # Count how many entries will be deleted
    entries_to_delete = 0
    for person in data.PEOPLE:
        entries_to_delete += len([entry for entry in data.watering_history[person] if entry.startswith(search_pattern)])
    
    if entries_to_delete == 0:
        messagebox.showinfo("Info", f"No matching entries found for {year_selection} {week_selection}.")
        return
    
    # Confirmation dialog
    if messagebox.askyesno("Confirm Delete", f"This will delete {entries_to_delete} entries for {year_selection} {week_selection}. Continue?"):
        for person in data.PEOPLE:
            data.watering_history[person] = [entry for entry in data.watering_history[person] if not entry.startswith(search_pattern)]

        # Save changes to JSON file
        save_to_file()

        # Excel functionality removed - using JSON-only data storage
        
        week_var.set("")
        manual_year_var.set("")
        update_people_list()
        update_schedule_display()
        update_status()
        # Update tabelle management displays
        if 'tabelle_manager' in globals():
            tabelle_manager.update_displays()
        messagebox.showinfo("Success", f"Entry for {year_selection} {week_selection} deleted successfully.")

def get_all_weeks_assignments():
    # Aggregate all week assignments from watering_history
    week_assignments = {}
    for person, entries in data.watering_history.items():
        for entry in entries:
            m = re.match(r"(\d{4} KW \d+): (.+) and (.+)", entry)
            if m:
                week, p1, p2 = m.group(1), m.group(2), m.group(3)
                week_assignments[week] = (p1, p2)
    # Sort by week number
    def week_sort_key(week):
        year, kw = re.match(r"(\d{4}) KW (\d+)", week).groups()
        return (int(year), int(kw))
    return [ (week, week_assignments[week][0], week_assignments[week][1]) 
             for week in sorted(week_assignments, key=week_sort_key) ]

# Initialize the GUI
def update_all_displays():
    """Update all display elements when data changes"""
    reload_current_data()  # Reload data from file to ensure we have the latest
    update_people_list()
    update_person_combos()
    update_schedule_display()
    update_status()
    # Update tabelle management displays
    if 'tabelle_manager' in globals():
        tabelle_manager.update_displays()

def initialize_gui():
    update_all_displays()
    refresh_years()  # Initialize year selection
    
    # Initialize backup system status
    if BACKUP_AVAILABLE:
        update_template_status()
    
    # Initialize manual management controls
    if 'manual_year_combo' in globals():
        available_years = data.get_available_years()
        manual_year_combo['values'] = available_years
        if available_years:
            manual_year_var.set(available_years[-1])  # Set to latest year
    
    # Bind double-click on treeview to populate name entry
    def on_tree_double_click(event):
        selection = people_tree.selection()
        if selection:
            name = people_tree.item(selection[0])['values'][0]
            name_entry.delete(0, tk.END)
            name_entry.insert(0, name)
    
    people_tree.bind('<Double-1>', on_tree_double_click)
    
    # Bind Enter key to add person
    def on_enter_key(event):
        add_person()
    
    name_entry.bind('<Return>', on_enter_key)

# Keyboard shortcuts
def setup_keyboard_shortcuts():
    root.bind('<Control-g>', lambda e: generate_and_show_schedule())
    root.bind('<Control-n>', lambda e: name_entry.focus())
    root.bind('<F1>', lambda e: show_help())

def show_help():
    help_text = """
    Gießplan Generator Help
    
    Keyboard Shortcuts:
    • Ctrl+G: Generate schedule
    • Ctrl+N: Focus name entry
    • F1: Show this help
    • Enter: Add person (when name entry is focused)
    
    Usage:
    1. Add people in the People Management tab
    2. Generate schedules in the Schedule Generation tab
    3. Manually manage specific dates in Manual Management tab
    
    Tips:
    • Double-click a person in the list to select them
    • The system automatically balances workload
    • Excel files are saved automatically
    • Red Cross Institute themed for professional use
    """
    messagebox.showinfo("Help", help_text)

# Initialize everything
initialize_gui()
setup_keyboard_shortcuts()

# Autofill person1 and person2 when week or year changes
def autofill_persons_for_week(*args):
    week_selection = week_var.get().strip()
    year_selection = manual_year_var.get().strip()
    if week_selection and year_selection:
        week_number = week_selection.replace("KW ", "").strip()
        existing_data = get_week_data_with_ersatz(year_selection, week_number)
        
        # existing_data format: [person1, person2, ersatz_person1, ersatz_person2]
        if existing_data and len(existing_data) >= 4:
            person1_var.set(existing_data[0] or "")
            person2_var.set(existing_data[1] or "")
            ersatz_person1_var.set(existing_data[2] or "")
            ersatz_person2_var.set(existing_data[3] or "")
        else:
            # If no data found, clear all fields
            person1_var.set("")
            person2_var.set("")
            ersatz_person1_var.set("")
            ersatz_person2_var.set("")

# Bind autofill to week and year changes
week_var.trace_add("write", autofill_persons_for_week)
manual_year_var.trace_add("write", autofill_persons_for_week)
week_combo.bind('<<ComboboxSelected>>', lambda e: autofill_persons_for_week())
manual_year_combo.bind('<<ComboboxSelected>>', lambda e: autofill_persons_for_week())

# Initialize combo boxes with current data
update_person_combos()

root.mainloop()
