import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

def flip_random_ones(n, m, num_ones, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    total_cells = n * m
    random_choice = rng.choice(total_cells, size=num_ones, replace=False)
    flat = np.zeros(total_cells, dtype=int)
    flat[random_choice] = 1
    return flat.reshape(n, m)

def classify_simulation(col_sums, thresholds):
    m = len(col_sums)
    row0 = thresholds[0, :]
    row1 = thresholds[1, :]

    all_below_row0 = all(col_sums[j] < row0[j] for j in range(m))
    all_below_row1 = all(col_sums[j] < row1[j] for j in range(m))
    any_at_least_row0 = any(col_sums[j] >= row0[j] for j in range(m))
    any_at_least_row1 = any(col_sums[j] >= row1[j] for j in range(m))

    # Case 2
    if all_below_row0 and all_below_row1:
        return 2
    # Case 4
    elif all_below_row0 and any_at_least_row1:
        return 4
    # Case 1
    elif any_at_least_row0 and all_below_row1:
        return 1
    # Case 3
    else:
        return 3

def append_c4_star2_sum_probability(case_probs):
    k_values = sorted(case_probs.keys())
    running_sum = 0.0
    for k in k_values:
        # case_probs[k] = [p1, p2, p3, p4, p4_star2, p4_star4]
        p4_star2_k = case_probs[k][4]  # The 5th element
        running_sum += p4_star2_k
        case_probs[k].append(running_sum)  # 7th element = cumulative sum
    return case_probs

# Optional: if you want to keep the plot function for later
def plot_case_probabilities(case_probs):
    """
    case_probs[k] = [p1, p2, p3, p4, p4_star2, p4_star4, p4_star2_sum].
    """
    k_values = sorted(case_probs.keys())
    c1_arr, c2_arr, c3_arr, c4_arr = [], [], [], []
    c4s2_arr, c4s4_arr, c4s2_sum_arr = [], [], []

    for k in k_values:
        p1, p2, p3, p4, p4s2, p4s4, p4s2_sum = case_probs[k]
        c1_arr.append(p1)
        c2_arr.append(p2)
        c3_arr.append(p3)
        c4_arr.append(p4)
        c4s2_arr.append(p4s2)
        c4s4_arr.append(p4s4)
        c4s2_sum_arr.append(p4s2_sum)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, c1_arr, marker='D', color='orange', label='Fall 1: P1(i)')
    ax.plot(k_values, c2_arr, marker='o', color='darkblue', label='Fall 2: P2(i)')
    ax.plot(k_values, c3_arr, marker='s', color='purple', label='Fall 3: P3(i)')
    ax.plot(k_values, c4_arr, marker='^', color='green', label='Fall 4: P4(i)')
    ax.plot(k_values, c4s2_arr, marker='^', color='blue', label='Fall 4*: P4*(i)')
    ax.plot(k_values, c4s2_sum_arr, marker='^', color='black', label='Fall 4* Sum: P4*(i)')

    ax.set_xlabel('Anzahl Spanngliedbrüche i')
    ax.set_ylabel('Auftretenswahrscheinlichkeit')
    ax.set_title('Fortschreitende Wahrscheinlichkeiten pro k')
    ax.set_yscale('log')
    ax.set_ylim(1e-9, 1.0)
    ax.set_yticks([1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    fig.tight_layout()
    return fig, ax

# =============== Main Streamlit App ===============
st.title("Stochastische Methode SpRK (Monte Carlo Simulation)")
st.write("Entsp. Lingemann, Zum Ankündigungsverhalten von älteren Brückenbauwerken bei Spannstahlausfällen infolge von Spannungsrisskorrosion, 2010, TUM")
st.write("Default Einstellungen Beispiel Bild 6.7")
# --- 1) Inputs: n, m ---
n = st.slider(
    "n (Anzahl Spannglieder)", 
    min_value=2, 
    max_value=10, 
    value=10,  # <-- set default here
    step=1
)
m = st.slider(
    "m (Anzahl Querschnitte)",
    min_value=2,
    max_value=10,
    value=5,   # <-- set default here
    step=1
)

# --- 2) num_sims ---
num_sims = st.select_slider(
    "Anzahl der Simulationen",
    options=[10_000, 100_000, 1_000_000, 10_000_000],
    format_func=lambda x: f"{x:,} Simulationen",
    value=100_000
)

# --- 3) thresholds (2 x m) ---
st.markdown("### Eingabe Anzahl Spanngliedausfall für Riss und Bruch")
default_row0 = "5,5,4,5,5"
default_row1 = "4,4,5,4,4"

row0_str = st.text_input(
    f"Anzahl Riss (m={m} Werte, Komma-getrennt)",
    value=default_row0
)
row1_str = st.text_input(
    f"Anzahl Bruch (m={m} Werte, Komma-getrennt)",
    value=default_row1
)

def parse_threshold_str(s, m):
    """Parse comma-separated floats from the user. 
       Ensure exactly m values are provided."""
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) != m:
        st.error(f"Bitte genau {m} Werte eingeben, gefunden: {len(vals)}")
        st.stop()
    return vals

try:
    # 1) Parse raw user input
    row0_raw = parse_threshold_str(row0_str, m)  # e.g., [5,5,4,5,5]
    row1_raw = parse_threshold_str(row1_str, m)  # e.g., [4,4,5,4,4]

    # 2) Convert them to thresholds via n - value
    row0_vals = [n - x for x in row0_raw]
    row1_vals = [n - x for x in row1_raw]

    # 3) Combine into a 2×m array
    thresholds = np.vstack([row0_vals, row1_vals])

except:
    st.stop()

st.write("Tabelle Spanngliedanzahl für Riss / Bruch je Querschnitt:")
st.write(thresholds)

# --- 4) Range of k ---
max_k = n*m
ones_range = range(0, max_k + 1)

# --- 5) Run button ---
if st.button("Starte Monte Carlo Simulation"):

    st.write("**Simulation läuft... bitte warten**")
    progress_bar = st.progress(0)
    st.write("Fall 1, Fall 2, Fall 3, Fall 4, Fall 4 aus Fall 2, Fall 4 aus Fall 4:")
    st.write("Fall 1: Mindestens ein gerissener QS, kein Versagen im QS")
    st.write("Fall 2: Kein gerissener QS, kein Versagen")
    st.write("Fall 3: Mind. ein gerissener QS, mind. ein Versagen im QS")
    st.write("Fall 4: Kein gerissener QS, mind. ein Versagen im QS")
    # For partial text output during the loop:
    output_placeholder = st.empty()

    case_counts = {}
    case_probs = {}
    case4_star_from2 = {}
    case4_star_from4 = {}

    for k_val in ones_range:
        case4_star_from2[k_val] = 0.0
        case4_star_from4[k_val] = 0.0

    rng = np.random.default_rng()
    total_k = len(ones_range)
    overall_start_time = time.time()

    # We'll collect lines of text in a list (for display) so it doesn't get overwritten each iteration
    results_text_lines = []

    # --- Main loop over k ---
    for idx, k in enumerate(ones_range, start=1):
        c1 = c2 = c3 = c4 = 0

        # Monte Carlo simulations for this k
        for _ in range(num_sims):
            X = flip_random_ones(n, m, k, rng=rng)
            col_sums = X.sum(axis=0)
            case_label = classify_simulation(col_sums, thresholds)

            if case_label == 1:
                c1 += 1
            elif case_label == 2:
                c2 += 1
            elif case_label == 3:
                c3 += 1
            elif case_label == 4:
                c4 += 1
                if k > 0:
                    ones_positions = np.argwhere(X == 1)
                    c2_sub = 0
                    c4_sub = 0
                    for (row_i, col_j) in ones_positions:
                        X_sub = X.copy()
                        X_sub[row_i, col_j] = 0
                        sub_sums = X_sub.sum(axis=0)
                        sub_case = classify_simulation(sub_sums, thresholds)
                        if sub_case == 2:
                            c2_sub += 1
                        elif sub_case == 4:
                            c4_sub += 1
                    fraction_from2 = c2_sub / k
                    fraction_from4 = c4_sub / k
                    case4_star_from2[k] += fraction_from2
                    case4_star_from4[k] += fraction_from4

        # Store final tallies for k
        case_counts[k] = [c1, c2, c3, c4, case4_star_from2[k], case4_star_from4[k]]

        # Compute probabilities for k
        c1p = c1 / num_sims
        c2p = c2 / num_sims
        c3p = c3 / num_sims
        c4p = c4 / num_sims
        c4s2p = case4_star_from2[k] / num_sims
        c4s4p = case4_star_from4[k] / num_sims
        case_probs[k] = [c1p, c2p, c3p, c4p, c4s2p, c4s4p]

        # Add partial text output line     
        line_counts = f"Fälle: k={k}: {case_counts[k]}"
        line_probs = (
            f"Wahrscheinlichkeiten (k={k}): "
            + ", ".join([f"{val:.3e}" for val in case_probs[k]])
        )
        results_text_lines.append(line_counts)
        results_text_lines.append(line_probs)

        # Re-render the text so far
        # (We join all lines so they stack in the output)
        output_placeholder.text("\n".join(results_text_lines))

        # Update progress bar
        progress_bar.progress(idx / total_k)

    # --- End of the main loop ---
    # Now we append the c4_star2_sum permanently
    case_probs = append_c4_star2_sum_probability(case_probs)

    overall_end_time = time.time()
    st.write(f"**Fertig!** Gesamtdauer: {overall_end_time - overall_start_time:.1f} Sekunden.")

    # Plot final probabilities
    fig, ax = plot_case_probabilities(case_probs)
    st.pyplot(fig)

    # Finally, the last k's cumulative sum
    final_prob_vector = case_probs[max_k]
    prob_case4s2_sum = final_prob_vector[-1]
    st.markdown(
        f"<h2 style='color:red;'>Versagenswahrscheinlichkeit: {prob_case4s2_sum:.6e}</h2>",
        unsafe_allow_html=True
    )
    st.success("Simulation abgeschlossen. Siehe Ergebnisse oben.")
