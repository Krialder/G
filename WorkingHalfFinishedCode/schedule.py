import random
import datetime
import json
import os
import re
import data
from data import save_to_file, reload_current_data
from tkinter import messagebox

def update_statistics():
    for person in data.PEOPLE:
        watering_count = len(data.watering_history.get(person, []))
        data.WEIGHTS[data.PEOPLE.index(person)] = max(1, 10 - watering_count)
    save_to_file()

def calculate_weighted_score(person_index, selection_count, total_weeks_active=None):
    """Calculate weighted arithmetic mean score for a person"""
    person = data.PEOPLE[person_index]
    base_weight = data.WEIGHTS[person_index]
    watering_count = len(data.watering_history.get(person, []))
    recent_selections = selection_count.get(person, 0)
    
    # Calculate how long this person has been in the system
    if total_weeks_active is None:
        total_weeks_active = max(1, len([entry for entries in data.watering_history.values() for entry in entries if "KW" in entry]) // len(data.PEOPLE))
    
    # Normalize watering count based on time active
    # For short-term people (< 4 weeks), give higher initial priority
    # For long-term people (> 52 weeks), use balanced approach
    if total_weeks_active <= 4:
        # Short-term: prioritize new people heavily
        fairness_factor = max(1, 5 - watering_count)
        time_factor = 2.0  # Boost for short-term people
    elif total_weeks_active <= 52:
        # Medium-term: balanced approach
        expected_waterings = total_weeks_active * 2 / len(data.PEOPLE)  # 2 people per week
        fairness_factor = max(0.5, expected_waterings - watering_count + 1)
        time_factor = 1.0
    else:
        # Long-term: focus on recent balance
        expected_waterings = total_weeks_active * 2 / len(data.PEOPLE)
        fairness_factor = max(0.5, expected_waterings - watering_count + 1)
        time_factor = 0.8  # Slight reduction for very long-term
    
    # Recent selection penalty (avoid consecutive selections)
    recent_penalty = max(0.1, 1.0 - (recent_selections * 0.3))
    
    # Weighted arithmetic mean calculation
    # Components: base_weight, fairness_factor, time_factor, recent_penalty
    score = (base_weight * 0.3) + (fairness_factor * 0.4) + (time_factor * 0.2) + (recent_penalty * 0.1)
    
    return max(0.1, score)  # Ensure minimum score

def select_with_smart_pairing(selection_count, total_weeks_active, new_people, experienced_people):
    """Smart pairing logic with multi-tiered priority system while maintaining weight-based fairness"""
    
    # Calculate scores for all people
    all_scores = []
    for i in range(len(data.PEOPLE)):
        score = calculate_weighted_score(i, selection_count, total_weeks_active)
        all_scores.append((data.PEOPLE[i], score))
    
    # Sort by score (highest first)
    all_scores.sort(key=lambda x: x[1], reverse=True)
    
    # If we have less than 2 people, fall back to regular selection
    if len(all_scores) < 2:
        return select_regular_two_people(all_scores, total_weeks_active)
    
    # Separate people by experience level and score
    people_by_experience = {
        'new': [],
        'beginner': [],
        'learning': [],
        'experienced': []
    }
    
    for person, score in all_scores:
        experience_level = data.get_person_experience_level(person)
        watering_count = len(data.watering_history.get(person, []))
        people_by_experience[experience_level].append((person, score, watering_count))
    
    # Sort each experience group by score (highest first)
    for level in people_by_experience:
        people_by_experience[level].sort(key=lambda x: x[1], reverse=True)
    
    # Multi-tiered selection strategy:
    # Priority 1: New people paired with experienced people (learning people also considered)
    # Priority 2: If no experienced people, new people paired with learning people
    # Priority 3: If only learning and experienced people, selection goes by weight alone
    
    selected_people = []
    
    # Check availability of different experience levels
    has_new_people = bool(people_by_experience['new'])
    has_experienced_people = bool(people_by_experience['experienced'])
    has_learning_people = bool(people_by_experience['learning'])
    
    # Priority 1: New people with experienced people available (learning people also considered)
    if has_new_people and has_experienced_people:
        # Select highest scoring new person
        new_person = people_by_experience['new'][0][0]
        selected_people.append(new_person)
        
        # Create combined pool of experienced and learning people for pairing
        experienced_and_learning = people_by_experience['experienced'] + people_by_experience['learning']
        # Sort the combined pool by score (highest first)
        experienced_and_learning.sort(key=lambda x: x[1], reverse=True)
        
        # Pair with highest scoring person from experienced/learning pool
        for person, score, watering_count in experienced_and_learning:
            if person != new_person:
                selected_people.append(person)
                break
    
    # Priority 2: New people but no experienced people available
    elif has_new_people and not has_experienced_people and has_learning_people:
        # Select highest scoring new person
        new_person = people_by_experience['new'][0][0]
        selected_people.append(new_person)
        
        # Pair with highest scoring learning person
        for person, score, watering_count in people_by_experience['learning']:
            if person != new_person:
                selected_people.append(person)
                break
    
    # Priority 3: Only learning and experienced people (no new people)
    elif not has_new_people and (has_learning_people or has_experienced_people):
        # Pure weight-based selection - take top 2 by score
        selected_people = [person for person, score in all_scores[:2]]
    
    # Handle edge cases where we still don't have 2 people
    if len(selected_people) < 2:
        # If we have new people but couldn't pair them properly
        if has_new_people:
            # Take highest scoring new person if not already selected
            if not selected_people:
                selected_people.append(people_by_experience['new'][0][0])
            
            # Try to find a partner from any category
            for person, score in all_scores:
                if person not in selected_people:
                    selected_people.append(person)
                    break
        else:
            # No new people - pure weight-based selection
            selected_people = [person for person, score in all_scores[:2]]
    
    return selected_people[:2]  # Ensure we return exactly 2 people

def select_with_dynamic_pairing(selection_count, total_weeks_active):
    """Dynamic pairing logic that prioritizes pure weight-based fairness with smart pairing preferences"""
    
    # Calculate base scores for all people - NO experience bonuses, pure weight-based
    all_scores = []
    for i in range(len(data.PEOPLE)):
        score = calculate_weighted_score(i, selection_count, total_weeks_active)
        person = data.PEOPLE[i]
        experience_level = data.get_person_experience_level(person)
        all_scores.append((person, score, experience_level))
    
    # Sort by score (highest first) - pure weight-based order
    all_scores.sort(key=lambda x: x[1], reverse=True)
    
    # If we have less than 2 people, fall back to regular selection
    if len(all_scores) < 2:
        return select_regular_two_people([(p, s) for p, s, e in all_scores], total_weeks_active)
    
    # Get top candidates based on weights only
    top_candidates = all_scores[:min(4, len(all_scores))]
    
    selected_people = []
    
    # Strategy: Use weighted selection, but if a new person gets selected,
    # try to pair them with experienced people when possible
    
    # First selection: Pure weight-based selection from top candidates
    candidates = [p for p, s, e in top_candidates]
    weights = [s for p, s, e in top_candidates]
    
    if candidates:
        first_person = random.choices(candidates, weights=weights, k=1)[0]
        first_person_experience = next((e for p, s, e in top_candidates if p == first_person), "experienced")
        selected_people.append(first_person)
        
        # Second selection: If first person is new, prefer experienced people for pairing
        remaining_candidates = [(p, s, e) for p, s, e in top_candidates if p != first_person]
        
        if first_person_experience == "new" and remaining_candidates:
            # Try to find experienced/learning people in remaining top candidates
            experienced_candidates = [(p, s) for p, s, e in remaining_candidates if e in ["experienced", "learning"]]
            
            if experienced_candidates and len(experienced_candidates) > 0:
                # Use weighted selection among experienced people
                exp_people = [p for p, s in experienced_candidates[:2]]  # Top 2 experienced
                exp_weights = [s for p, s in experienced_candidates[:2]]
                if exp_people:
                    second_person = random.choices(exp_people, weights=exp_weights, k=1)[0]
                    selected_people.append(second_person)
            else:
                # No experienced people available, use regular weight-based selection
                if remaining_candidates:
                    remaining_people = [p for p, s, e in remaining_candidates[:2]]
                    remaining_weights = [s for p, s, e in remaining_candidates[:2]]
                    if remaining_people:
                        second_person = random.choices(remaining_people, weights=remaining_weights, k=1)[0]
                        selected_people.append(second_person)
        else:
            # First person is not new, OR no experienced people available
            # Use pure weight-based selection for second person
            if remaining_candidates:
                remaining_people = [p for p, s, e in remaining_candidates[:2]]
                remaining_weights = [s for p, s, e in remaining_candidates[:2]]
                if remaining_people:
                    second_person = random.choices(remaining_people, weights=remaining_weights, k=1)[0]
                    selected_people.append(second_person)
    
    # Fallback to regular selection if needed
    if len(selected_people) < 2:
        return select_regular_two_people([(p, s) for p, s, e in all_scores], total_weeks_active)
    
    return selected_people[:2]  # Ensure we return exactly 2 people

def select_regular_two_people(all_scores, total_weeks_active):
    """Regular selection logic for two people"""
    if total_weeks_active <= 4:
        # Short-term: pick top 2 directly with slight randomness
        top_candidates = all_scores[:min(3, len(all_scores))]
        candidates = [person for person, score in top_candidates]
        weights = [score for person, score in top_candidates]
        
        selected = []
        remaining_candidates = candidates.copy()
        remaining_weights = weights.copy()
        
        for _ in range(min(2, len(remaining_candidates))):
            if not remaining_candidates:
                break
            chosen = random.choices(remaining_candidates, weights=remaining_weights, k=1)[0]
            selected.append(chosen)
            idx = remaining_candidates.index(chosen)
            remaining_candidates.pop(idx)
            remaining_weights.pop(idx)
    else:
        # Long-term: use weighted selection from top candidates
        top_candidates = all_scores[:min(4, len(all_scores))]
        candidates = [person for person, score in top_candidates]
        weights = [score for person, score in top_candidates]
        
        selected = []
        remaining_candidates = candidates.copy()
        remaining_weights = weights.copy()
        
        for _ in range(min(2, len(remaining_candidates))):
            if not remaining_candidates:
                break
            chosen = random.choices(remaining_candidates, weights=remaining_weights, k=1)[0]
            selected.append(chosen)
            idx = remaining_candidates.index(chosen)
            remaining_candidates.pop(idx)
            remaining_weights.pop(idx)
    
    return selected

def select_people_weighted_mean(selection_count, current_week_in_year=None):
    """Select 2 people using weighted arithmetic mean approach with dynamic pairing"""
    # Calculate total weeks active in system - filter out non-list entries
    all_week_entries = []
    for entries in data.watering_history.values():
        if isinstance(entries, list):
            for entry in entries:
                if "KW" in entry:
                    all_week_entries.append(entry)
    
    # Calculate base total weeks active from existing history
    base_total_weeks_active = len(set(all_week_entries)) if all_week_entries else 1
    
    # If we're in the middle of generating a schedule, adjust total_weeks_active
    # to reflect the progress we've made in the current generation
    if current_week_in_year is not None:
        # Use the maximum of existing history or current week position
        total_weeks_active = max(base_total_weeks_active, current_week_in_year)
    else:
        total_weeks_active = base_total_weeks_active
    
    # Use dynamic pairing logic - experience-based but not fixed pairs
    return select_with_dynamic_pairing(selection_count, total_weeks_active)

def calculate_weighted_score_extra(person_index, selection_count, total_weeks_active=None):
    """Calculate weighted arithmetic mean score for a person using extra weights"""
    person = data.PEOPLE[person_index]
    base_weight = data.EXTRA_WEIGHTS[person_index] if person_index < len(data.EXTRA_WEIGHTS) else 1
    watering_count = len(data.watering_history.get(person, []))
    recent_selections = selection_count.get(person, 0)
    
    # Calculate how long this person has been in the system
    if total_weeks_active is None:
        total_weeks_active = max(1, len([entry for entries in data.watering_history.values() for entry in entries if "KW" in entry]) // len(data.PEOPLE))
    
    # Same logic as regular selection but using extra weights
    if total_weeks_active <= 4:
        fairness_factor = max(1, 5 - watering_count)
        time_factor = 2.0
    elif total_weeks_active <= 52:
        expected_waterings = total_weeks_active * 2 / len(data.PEOPLE)
        fairness_factor = max(0.5, expected_waterings - watering_count + 1)
        time_factor = 1.0
    else:
        expected_waterings = total_weeks_active * 2 / len(data.PEOPLE)
        fairness_factor = max(0.5, expected_waterings - watering_count + 1)
        time_factor = 0.8
    
    # Recent selection penalty
    recent_penalty = max(0.1, 1.0 - (recent_selections * 0.3))
    
    # Weighted arithmetic mean calculation using extra weights
    score = (base_weight * 0.3) + (fairness_factor * 0.4) + (time_factor * 0.2) + (recent_penalty * 0.1)
    
    return max(0.1, score)

def select_ersatz_people_weighted_mean(selection_count, excluded_persons=None, current_week_in_year=None):
    """Select 2 ErsatzPersons using weighted arithmetic mean approach with extra weights and smart pairing
    
    Args:
        selection_count: Dictionary tracking how many times each person has been selected
        excluded_persons: List of persons to exclude from ersatz selection (main persons)
        current_week_in_year: Current week in the year to adjust total_weeks_active calculation
    """
    if excluded_persons is None:
        excluded_persons = []
    
    # Calculate total weeks active in system
    all_week_entries = []
    for entries in data.watering_history.values():
        if isinstance(entries, list):
            for entry in entries:
                if "KW" in entry:
                    all_week_entries.append(entry)
    
    # Calculate base total weeks active from existing history
    base_total_weeks_active = len(set(all_week_entries)) if all_week_entries else 1
    
    # If we're in the middle of generating a schedule, adjust total_weeks_active
    # to reflect the progress we've made in the current generation
    if current_week_in_year is not None:
        # Use the maximum of existing history or current week position
        total_weeks_active = max(base_total_weeks_active, current_week_in_year)
    else:
        total_weeks_active = base_total_weeks_active
    
    # Get available people (excluding main persons)
    available_people = [person for person in data.PEOPLE if person not in excluded_persons]
    
    # Get experience levels for smart pairing among available people
    available_new_people = [person for person in available_people if data.get_person_experience_level(person) in ["new", "beginner"]]
    available_experienced_people = [person for person in available_people if data.get_person_experience_level(person) == "experienced"]
    
    scores = []
    for i in range(len(data.PEOPLE)):
        # Skip persons who are already selected as main persons
        if data.PEOPLE[i] in excluded_persons:
            continue
        score = calculate_weighted_score_extra(i, selection_count, total_weeks_active)
        scores.append((data.PEOPLE[i], score))
    
    # Sort by score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # If there are new people available, try to pair them with experienced people
    if available_new_people and available_experienced_people:
        # Find the highest scoring new person
        top_new_person = None
        for person, score in scores:
            if person in available_new_people:
                top_new_person = person
                break
        
        # Find the best experienced person to pair with
        best_experienced_person = None
        for person, score in scores:
            if person in available_experienced_people:
                best_experienced_person = person
                break
        
        if top_new_person and best_experienced_person:
            return [top_new_person, best_experienced_person]
    
    # Use dynamic pairing logic for ersatz selection too
    return select_dynamic_ersatz_pairing(scores, total_weeks_active)

def select_regular_two_people_from_scores(scores, total_weeks_active):
    """Regular selection logic for two people from pre-calculated scores"""
    if total_weeks_active <= 4:
        # Short-term: pick top 2 directly with slight randomness
        top_candidates = scores[:min(3, len(scores))]
        candidates = [person for person, score in top_candidates]
        weights = [score for person, score in top_candidates]
        
        selected = []
        remaining_candidates = candidates.copy()
        remaining_weights = weights.copy()
        
        for _ in range(min(2, len(remaining_candidates))):
            if not remaining_candidates:
                break
            chosen = random.choices(remaining_candidates, weights=remaining_weights, k=1)[0]
            selected.append(chosen)
            idx = remaining_candidates.index(chosen)
            remaining_candidates.pop(idx)
            remaining_weights.pop(idx)
    else:
        # Long-term: use weighted selection from top candidates
        top_candidates = scores[:min(4, len(scores))]
        candidates = [person for person, score in top_candidates]
        weights = [score for person, score in top_candidates]
        
        selected = []
        remaining_candidates = candidates.copy()
        remaining_weights = weights.copy()
        
        for _ in range(min(2, len(remaining_candidates))):
            if not remaining_candidates:
                break
            chosen = random.choices(remaining_candidates, weights=remaining_weights, k=1)[0]
            selected.append(chosen)
            idx = remaining_candidates.index(chosen)
            remaining_candidates.pop(idx)
            remaining_weights.pop(idx)
    
    return selected

def select_dynamic_ersatz_pairing(scores, total_weeks_active):
    """Dynamic pairing logic for ersatz selection - pure weight-based with smart pairing preferences"""
    
    # If we have fewer than 2 people available, use regular selection
    if len(scores) < 2:
        return select_regular_two_people_from_scores(scores, total_weeks_active)
    
    # Get experience levels for available people - NO score bonuses, pure weight-based
    scores_with_experience = []
    for person, score in scores:
        experience_level = data.get_person_experience_level(person)
        scores_with_experience.append((person, score, experience_level))
    
    # Sort by score (highest first) - pure weight-based order
    scores_with_experience.sort(key=lambda x: x[1], reverse=True)
    
    # Get top candidates
    top_candidates = scores_with_experience[:min(4, len(scores_with_experience))]
    
    selected_people = []
    
    # First selection: Pure weight-based selection from top candidates
    candidates = [p for p, s, e in top_candidates]
    weights = [s for p, s, e in top_candidates]
    
    if candidates:
        first_person = random.choices(candidates, weights=weights, k=1)[0]
        first_person_experience = next((e for p, s, e in top_candidates if p == first_person), "experienced")
        selected_people.append(first_person)
        
        # Second selection: If first person is new, prefer experienced people for pairing
        remaining_candidates = [(p, s, e) for p, s, e in top_candidates if p != first_person]
        
        if first_person_experience == "new" and remaining_candidates:
            # Try to find experienced/learning people in remaining top candidates
            experienced_candidates = [(p, s) for p, s, e in remaining_candidates if e in ["experienced", "learning"]]
            
            if experienced_candidates:
                # Use weighted selection among experienced people
                exp_people = [p for p, s in experienced_candidates[:2]]
                exp_weights = [s for p, s in experienced_candidates[:2]]
                if exp_people:
                    second_person = random.choices(exp_people, weights=exp_weights, k=1)[0]
                    selected_people.append(second_person)
            else:
                # No experienced people available, use regular weight-based selection
                remaining_people = [p for p, s, e in remaining_candidates[:2]]
                remaining_weights = [s for p, s, e in remaining_candidates[:2]]
                if remaining_people:
                    second_person = random.choices(remaining_people, weights=remaining_weights, k=1)[0]
                    selected_people.append(second_person)
        else:
            # First person is not new, use pure weight-based selection for second person
            remaining_people = [p for p, s, e in remaining_candidates[:2]]
            remaining_weights = [s for p, s, e in remaining_candidates[:2]]
            if remaining_people:
                second_person = random.choices(remaining_people, weights=remaining_weights, k=1)[0]
                selected_people.append(second_person)
    
    # Final fallback
    if len(selected_people) < 2:
        return select_regular_two_people_from_scores(scores, total_weeks_active)
    
    return selected_people[:2]

def generate_schedule(schedule_type="Next 6 Weeks"):
    # Reload data to ensure we're using the most current file
    reload_current_data()
    
    update_statistics()
    full_schedule = []  # Keep track of all generated weeks
    new_weeks_only = []  # Only the newly generated weeks
    selection_count = {person: len(data.watering_history[person]) for person in data.PEOPLE}
    current_year = datetime.date.today().year
    current_week = datetime.date.today().isocalendar()[1]
    
    # Determine which year we're currently working in based on the data file
    if data.FILE_PATH.endswith("people.json"):
        schedule_year = current_year  # 2025
    else:
        # Extract year from filename like "people_2026.json"
        match = re.search(r'people_(\d{4})\.json', data.FILE_PATH)
        schedule_year = int(match.group(1)) if match else current_year

    # Filter out non-list entries when checking for week entries
    week_entries_found = []
    for person in data.watering_history.values():
        if isinstance(person, list):
            for entry in person:
                if "KW" in entry:
                    week_entries_found.append(entry)
    
    if week_entries_found:
        # Extract week numbers from KW entries (e.g., "2025 KW 15: ..." -> 15)
        week_numbers = []
        for entry in week_entries_found:
            if "KW" in entry:
                try:
                    # Find KW pattern and extract number
                    match = re.search(r'KW (\d+)', entry)
                    if match:
                        week_numbers.append(int(match.group(1)))
                except:
                    pass
        
        if week_numbers:
            last_week = max(week_numbers)
            # If last week is 52 or higher, start from week 1 of next year
            if last_week >= 52:
                if schedule_type == "Next 6 Weeks":
                    # Ask user if they want to create a new year
                    response = messagebox.askyesno("Year Complete", 
                                                f"Year {schedule_year} is complete (week {last_week} was the last week).\n\n"
                                                f"Do you want to create a new year {schedule_year + 1} and generate 6 weeks there?")
                    if response:
                        start_week = 1
                        schedule_year = schedule_year + 1
                        
                        # Create the new year file if it doesn't exist
                        new_year_file = f"people_{schedule_year}.json"
                        if not os.path.exists(new_year_file):
                            try:
                                # Create new year file with current people but empty watering history
                                new_year_data = {
                                    "PEOPLE": data.PEOPLE[:],  # Copy current people
                                    "WEIGHTS": data.WEIGHTS[:],  # Copy current weights
                                    "WATERING_HISTORY": {person: [] for person in data.PEOPLE}  # Empty history for new year
                                }
                                
                                # Try to write the file with better error handling
                                try:
                                    with open(new_year_file, "w") as file:
                                        json.dump(new_year_data, file, indent=2)
                                except PermissionError:
                                    messagebox.showerror("File Permission Error", 
                                                       f"Cannot create {new_year_file} - file may be open in another application.\n\n"
                                                       f"Please close any Excel files or other applications using this file and try again.")
                                    return []
                                except Exception as file_error:
                                    messagebox.showerror("File Creation Error", 
                                                       f"Failed to create {new_year_file}: {str(file_error)}")
                                    return []
                                    
                                print(f"Created new year file: {new_year_file}")
                            except Exception as e:
                                messagebox.showerror("Error", f"Failed to create new year file: {str(e)}")
                                return []
                        
                        # Load the new year data
                        if not data.load_year_data(schedule_year):
                            messagebox.showerror("Error", f"Failed to load new year data for {schedule_year}")
                            return []
                    else:
                        # User cancelled, return empty schedule
                        return []
                else:
                    # For "Remaining Weeks", if we're already at week 52, there are no remaining weeks
                    start_week = 53  # This will result in 0 weeks to generate
            else:
                start_week = last_week + 1
        else:
            start_week = current_week  # Changed from current_week + 1 to start from current week
    else:
        # No existing entries - this is first time use, start from current week
        start_week = current_week

    max_week = 52
    week = start_week
    year_transition_occurred = False  # Initialize for both schedule types
    
    # Calculate number of weeks to generate based on schedule type
    if schedule_type == "Remaining Weeks":
        # Calculate remaining weeks in current year
        if week <= max_week:
            weeks_to_generate = max_week - week + 1
        else:
            weeks_to_generate = 0  # No remaining weeks if we're already past week 52
            messagebox.showinfo("Year Complete", f"Year {schedule_year} is already complete. No remaining weeks to generate.")
            return []
        
        # Generate schedule for the calculated number of weeks
        for _ in range(weeks_to_generate):
            # Handle year transition
            if week > max_week:
                year_transition_occurred = True
                
                # Save only the new weeks generated in the current year
                if new_weeks_only:
                    # Excel functionality removed - using JSON-only data storage
                    pass
                
                # Transition to new year
                schedule_year = schedule_year + 1  # Increment from current year
                week = 1
                
                # Create new year data file and reset history
                new_json_file = f"people_{schedule_year}.json"
                new_history = {person: [] for person in data.PEOPLE}
                
                # Save new year file
                with open(new_json_file, "w") as file:
                    json.dump({"PEOPLE": data.PEOPLE, "WEIGHTS": data.WEIGHTS, "WATERING_HISTORY": new_history}, file)
                
                # Update global variables in data module
                data.FILE_PATH = new_json_file
                data.watering_history = new_history
                
                # Reset selection count for new year
                selection_count = {person: 0 for person in data.PEOPLE}
                
                # Excel functionality removed - using JSON-only data storage
                new_weeks_only = []  # Reset new weeks for new year

            # Use weighted arithmetic mean selection
            selected = select_people_weighted_mean(selection_count, current_week_in_year=week)
            ersatz_selected = select_ersatz_people_weighted_mean(selection_count, excluded_persons=selected, current_week_in_year=week)

            for person in selected:
                selection_count[person] += 1
                data.watering_history[person].append(f"{schedule_year} KW {week}: {selected[0]} and {selected[1]} (ErsatzPersons: {ersatz_selected[0]} and {ersatz_selected[1]})")

            week_entry = f"{schedule_year} KW {week}: {selected[0]} and {selected[1]} (ErsatzPersons: {ersatz_selected[0]} and {ersatz_selected[1]})"
            new_weeks_only.append(week_entry)
            full_schedule.append(week_entry)
            week += 1
    
    else:  # "Next 6 Weeks"
        weeks_to_generate = 6
        
        # Check if we'll cross a year boundary within the next 6 weeks
        weeks_remaining_in_year = max_week - week + 1
        
        # If we're already at or past week 52, or if we need to cross year boundary
        if weeks_remaining_in_year <= 0 or weeks_remaining_in_year < weeks_to_generate:
            year_transition_occurred = True
            
            # If we're already past week 52, skip to new year directly
            if weeks_remaining_in_year <= 0:
                year_transition_message = f"Year transition detected:\n\n"
                year_transition_message += f"Current year {schedule_year - 1} is already complete.\n"
                year_transition_message += f"Starting new year {schedule_year} with 6 weeks...\n"
                
                # We should already be in the new year due to the logic above
                # Generate all 6 weeks for the new year
                for _ in range(weeks_to_generate):
                    # Use weighted arithmetic mean selection
                    selected = select_people_weighted_mean(selection_count, current_week_in_year=week)
                    ersatz_selected = select_ersatz_people_weighted_mean(selection_count, excluded_persons=selected, current_week_in_year=week)

                    for person in selected:
                        selection_count[person] += 1
                        data.watering_history[person].append(f"{schedule_year} KW {week}: {selected[0]} and {selected[1]} (ErsatzPersons: {ersatz_selected[0]} and {ersatz_selected[1]})")

                    week_entry = f"{schedule_year} KW {week}: {selected[0]} and {selected[1]} (ErsatzPersons: {ersatz_selected[0]} and {ersatz_selected[1]})"
                    new_weeks_only.append(week_entry)
                    full_schedule.append(week_entry)
                    week += 1
                
                year_transition_message += f"✓ Generated 6 weeks for new year {schedule_year} (KW 1-6)\n"
                messagebox.showinfo("Year Transition Complete", year_transition_message)
                
            else:
                # We need to finish current year first, then start new year
                year_transition_message = f"Year transition detected:\n\n"
                year_transition_message += f"Step 1: Completing current year {schedule_year} with {weeks_remaining_in_year} weeks remaining...\n"
                
                current_year_weeks = []  # Track weeks for current year
                current_year_start_week = week  # Remember the starting week for this year
                
                # Generate ALL remaining weeks for the current year
                for _ in range(weeks_remaining_in_year):
                    # Use weighted arithmetic mean selection
                    selected = select_people_weighted_mean(selection_count, current_week_in_year=week)
                    ersatz_selected = select_ersatz_people_weighted_mean(selection_count, excluded_persons=selected, current_week_in_year=week)

                    for person in selected:
                        selection_count[person] += 1
                        data.watering_history[person].append(f"{schedule_year} KW {week}: {selected[0]} and {selected[1]} (ErsatzPersons: {ersatz_selected[0]} and {ersatz_selected[1]})")

                    week_entry = f"{schedule_year} KW {week}: {selected[0]} and {selected[1]} (ErsatzPersons: {ersatz_selected[0]} and {ersatz_selected[1]})"
                    current_year_weeks.append(week_entry)
                    full_schedule.append(week_entry)
                    week += 1
                
                # IMMEDIATELY save the completed year's schedule and data
                if current_year_weeks:
                    # Excel functionality removed - using JSON-only data storage
                    save_to_file()  # Save current year's data to JSON immediately
                current_year_end_week = week - 1
                if current_year_start_week == current_year_end_week:
                    year_transition_message += f"✓ Completed and saved 1 week for year {schedule_year} (KW {current_year_start_week})\n"
                else:
                    year_transition_message += f"✓ Completed and saved {len(current_year_weeks)} weeks for year {schedule_year} (KW {current_year_start_week}-{current_year_end_week})\n"
            
            year_transition_message += f"✓ Year {schedule_year} is now complete and saved.\n"
            
            # Step 2: NOW transition to new year
            original_schedule_year = schedule_year
            schedule_year = schedule_year + 1
            week = 1
            
            year_transition_message += f"\nStep 2: Now transitioning to new year {schedule_year}...\n"
            
            # Create new year data file and reset history
            new_json_file = f"people_{schedule_year}.json"
            new_history = {person: [] for person in data.PEOPLE}
            
            # Save new year file
            with open(new_json_file, "w") as file:
                json.dump({"PEOPLE": data.PEOPLE, "WEIGHTS": data.WEIGHTS, "WATERING_HISTORY": new_history}, file)
            
            # Update global variables in data module
            data.FILE_PATH = new_json_file
            data.watering_history = new_history
            
            # Reset selection count for new year
            selection_count = {person: 0 for person in data.PEOPLE}
            
            # Excel functionality removed - using JSON-only data storage
            
            # Step 3: Generate remaining weeks for the new year
            remaining_weeks_needed = weeks_to_generate - weeks_remaining_in_year
            year_transition_message += f"✓ Created new year file: people_{schedule_year}.json\n"
            year_transition_message += f"\nStep 3: Generating {remaining_weeks_needed} weeks for new year {schedule_year}...\n"
            
            new_year_weeks = []  # Track weeks for new year
            new_year_start_week = week  # Should be 1 for new year
            
            for _ in range(remaining_weeks_needed):
                # Use weighted arithmetic mean selection
                selected = select_people_weighted_mean(selection_count, current_week_in_year=week)
                ersatz_selected = select_ersatz_people_weighted_mean(selection_count, excluded_persons=selected, current_week_in_year=week)

                for person in selected:
                    selection_count[person] += 1
                    data.watering_history[person].append(f"{schedule_year} KW {week}: {selected[0]} and {selected[1]} (ErsatzPersons: {ersatz_selected[0]} and {ersatz_selected[1]})")

                week_entry = f"{schedule_year} KW {week}: {selected[0]} and {selected[1]} (ErsatzPersons: {ersatz_selected[0]} and {ersatz_selected[1]})"
                new_year_weeks.append(week_entry)
                full_schedule.append(week_entry)
                week += 1
            
            # Save the new year's schedule
            if new_year_weeks:
                # Excel functionality removed - using JSON-only data storage
                save_to_file()  # Save new year's data to JSON
                new_year_end_week = week - 1
                if new_year_start_week == new_year_end_week:
                    year_transition_message += f"✓ Generated and saved 1 week for new year {schedule_year} (KW {new_year_start_week})\n"
                else:
                    year_transition_message += f"✓ Generated and saved {len(new_year_weeks)} weeks for new year {schedule_year} (KW {new_year_start_week}-{new_year_end_week})\n"
            
            # Combine all weeks for the final list
            new_weeks_only = current_year_weeks + new_year_weeks
            
            year_transition_message += f"\nYear transition completed successfully!"
            
            # Create summary of all generated weeks
            total_weeks_summary = f"Total weeks generated: {len(new_weeks_only)}\n"
            if current_year_weeks:
                current_year_start = current_year_start_week
                current_year_end = current_year_start + len(current_year_weeks) - 1
                if len(current_year_weeks) == 1:
                    total_weeks_summary += f"• {original_schedule_year}: 1 week (KW {current_year_start})\n"
                else:
                    total_weeks_summary += f"• {original_schedule_year}: {len(current_year_weeks)} weeks (KW {current_year_start}-{current_year_end})\n"
            
            if new_year_weeks:
                if len(new_year_weeks) == 1:
                    total_weeks_summary += f"• {schedule_year}: 1 week (KW {new_year_start_week})\n"
                else:
                    total_weeks_summary += f"• {schedule_year}: {len(new_year_weeks)} weeks (KW {new_year_start_week}-{new_year_end_week})\n"
            
            year_transition_message += f"\n{total_weeks_summary}"
            
            # Show the transition summary
            messagebox.showinfo("Year Transition Complete", year_transition_message)
        
        else:
            # No year boundary crossing, generate normally
            year_transition_occurred = False
            start_week_for_message = week
            
            for _ in range(weeks_to_generate):
                # Use weighted arithmetic mean selection
                selected = select_people_weighted_mean(selection_count, current_week_in_year=week)
                ersatz_selected = select_ersatz_people_weighted_mean(selection_count, excluded_persons=selected, current_week_in_year=week)

                for person in selected:
                    selection_count[person] += 1
                    data.watering_history[person].append(f"{schedule_year} KW {week}: {selected[0]} and {selected[1]} (ErsatzPersons: {ersatz_selected[0]} and {ersatz_selected[1]})")

                week_entry = f"{schedule_year} KW {week}: {selected[0]} and {selected[1]} (ErsatzPersons: {ersatz_selected[0]} and {ersatz_selected[1]})"
                new_weeks_only.append(week_entry)
                full_schedule.append(week_entry)
                week += 1

    # Only save if we didn't already save during year transition
    if not year_transition_occurred:
        save_to_file()
    
    # Save only the newly generated weeks to Excel (only if not already saved during year transition)
    if new_weeks_only and not year_transition_occurred:
        # Excel functionality removed - using JSON-only data storage
        pass
    
    return full_schedule

def show_schedule(schedule_type="Next 6 Weeks"):
    # Reload current data before generating schedule
    reload_current_data()
    schedule = generate_schedule(schedule_type)
    result = "\n".join(schedule)
    messagebox.showinfo("Gießplan", result)
