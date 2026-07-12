import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def analyze_pendulum(filename, L, d):
    fs, data = wavfile.read(filename)

    if len(data.shape) > 1:
        data = data[:, 0]

    data = data / np.max(np.abs(data))

    peaks, _ = find_peaks(data, height=0.1, distance=int(0.3 * fs))

    t_peaks = peaks / fs
    dt = np.diff(t_peaks)
    mean_dt = np.mean(dt)

    T_raw = 2 * mean_dt
    omega = 2 * np.pi / T_raw
    dt_d = d / (omega * L)

    T = 2 * (mean_dt - dt_d)
    g = 4 * np.pi**2 * L / T**2

    return peaks, data, fs, T, g


class PendulumApp:

    def __init__(self, root):
        self.root = root
        root.title("Optical Pendulum Gravity Lab")
        root.geometry("1200x750")
        root.configure(bg="#f5f5f5")
        root.resizable(False, False)

        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.filename = None

        # LEFT PANEL
        left = tk.Frame(root, bg="white", highlightthickness=1, highlightbackground="#d0d0d0")
        left.pack(side="left", fill="y", padx=15, pady=15)

        # Header
        tk.Label(left, text="Optical Pendulum Gravity Lab",
                 font=("Arial", 16, "bold"),
                 fg="#1a1a1a",
                 bg="white").pack(pady=(20, 30))

        # Input section
        input_frame = tk.Frame(left, bg="white")
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="Parameters",
                 font=("Arial", 11, "bold"),
                 fg="#333333",
                 bg="white").pack(anchor="w", padx=20, pady=(0, 15))

        self.length_entry = self.create_entry(input_frame, "Pendulum Length (m)")
        self.diameter_entry = self.create_entry(input_frame, "Bob Diameter (m)")

        # File button
        self.file_btn = tk.Button(left, text="Select WAV File",
                                   command=self.browse_file,
                                   bg="#2196F3", fg="white",
                                   font=("Arial", 10, "bold"),
                                   relief="flat", width=30, height=2,
                                   cursor="hand2",
                                   activebackground="#1976D2")
        self.file_btn.pack(pady=15)

        # Run button
        self.run_btn = tk.Button(left, text="Run Analysis",
                                  command=self.run_analysis,
                                  bg="#4CAF50", fg="white",
                                  font=("Arial", 11, "bold"),
                                  relief="flat", width=30, height=2,
                                  cursor="hand2",
                                  activebackground="#45a049")
        self.run_btn.pack(pady=10)

        # Results section
        results_frame = tk.Frame(left, bg="#fafafa", highlightthickness=1, highlightbackground="#e0e0e0")
        results_frame.pack(pady=30, padx=15, fill="x")

        tk.Label(results_frame, text="Results",
                 font=("Arial", 10, "bold"),
                 fg="#555555",
                 bg="#fafafa").pack(pady=(15, 10))

        self.g_label = tk.Label(results_frame, text="g = ---",
                                font=("Arial", 28, "bold"),
                                fg="#2196F3",
                                bg="#fafafa")
        self.g_label.pack(pady=(5, 0))

        tk.Label(results_frame, text="m/s²",
                 font=("Arial", 11),
                 fg="#666666",
                 bg="#fafafa").pack()

        separator = tk.Frame(results_frame, bg="#e0e0e0", height=1)
        separator.pack(fill="x", padx=30, pady=15)

        self.error_label = tk.Label(results_frame, text="Error: --- %",
                                    font=("Arial", 12, "bold"),
                                    fg="#FF9800",
                                    bg="#fafafa")
        self.error_label.pack(pady=5)

        self.period_label = tk.Label(results_frame, text="Period: --- s",
                                     font=("Arial", 11),
                                     fg="#666666",
                                     bg="#fafafa")
        self.period_label.pack(pady=(5, 20))

        # RIGHT PANEL
        right = tk.Frame(root, bg="#f5f5f5")
        right.pack(side="right", fill="both", expand=True, padx=15, pady=15)

        # Graph title
        tk.Label(right, text="Waveform Analysis",
                 font=("Arial", 13, "bold"),
                 fg="#333333",
                 bg="#f5f5f5").pack(pady=(0, 10))

        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.patch.set_facecolor("#f5f5f5")
        self.ax.set_facecolor("white")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Initial empty plot
        self.ax.text(0.5, 0.5, "No data loaded\n\nSelect a WAV file to begin",
                     ha="center", va="center",
                     fontsize=13, color="#999999",
                     transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def create_entry(self, parent, placeholder):
        frame = tk.Frame(parent, bg="white", highlightthickness=1, highlightbackground="#d0d0d0")
        frame.pack(pady=8, padx=20, fill="x")

        entry = tk.Entry(frame, font=("Arial", 10),
                         bg="white", fg="#666666",
                         insertbackground="#2196F3",
                         relief="flat", bd=0)
        entry.insert(0, placeholder)
        entry.pack(fill="x", padx=10, pady=10)

        def on_focus_in(event):
            if entry.get() == placeholder:
                entry.delete(0, tk.END)
                entry.config(fg="#000000")
                frame.config(highlightbackground="#2196F3", highlightthickness=2)

        def on_focus_out(event):
            if entry.get() == "":
                entry.insert(0, placeholder)
                entry.config(fg="#666666")
            frame.config(highlightbackground="#d0d0d0", highlightthickness=1)

        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

        return entry

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filename:
            self.filename = filename
            short_name = filename.split('/')[-1]
            if len(short_name) > 25:
                short_name = short_name[:22] + "..."
            self.file_btn.config(text=f"✓ {short_name}", bg="#1976D2")

    def run_analysis(self):
        try:
            if not self.filename:
                messagebox.showwarning("Warning", "Please select a WAV file first.")
                return

            L_text = self.length_entry.get()
            d_text = self.diameter_entry.get()

            if "Pendulum Length" in L_text or "Bob Diameter" in d_text:
                messagebox.showwarning("Warning", "Please enter valid parameters.")
                return

            L = float(L_text)
            d = float(d_text)

            peaks, data, fs, T, g = analyze_pendulum(self.filename, L, d)

            g_true = 9.81
            error = abs(g - g_true) / g_true * 100

            if error < 1:
                color = "#4CAF50"
            elif error < 3:
                color = "#FF9800"
            else:
                color = "#F44336"

            self.g_label.config(text=f"{g:.4f}")
            self.error_label.config(text=f"Error: {error:.2f} %", fg=color)
            self.period_label.config(text=f"Period: {T:.4f} s")

            # Plot
            self.ax.clear()
            self.ax.set_facecolor("white")

            t = np.arange(len(data)) / fs
            self.ax.plot(t, data, color="#2196F3", linewidth=1.5, alpha=0.9, label="Signal")
            self.ax.scatter(peaks/fs, data[peaks], color="#F44336", s=60, 
                           zorder=5, edgecolors="white", linewidths=1.5, label="Peaks")

            self.ax.set_xlabel("Time (s)", color="#333333", fontsize=11)
            self.ax.set_ylabel("Amplitude", color="#333333", fontsize=11)
            self.ax.tick_params(colors="#666666", labelsize=9)
            self.ax.grid(alpha=0.3, color="#cccccc", linestyle="--", linewidth=0.8)
            self.ax.legend(loc="upper right", facecolor="white", 
                          edgecolor="#d0d0d0", labelcolor="#333333")

            for spine in self.ax.spines.values():
                spine.set_color("#d0d0d0")

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_close(self):
        plt.close("all")
        self.root.destroy()
        exit()


if __name__ == "__main__":
    root = tk.Tk()
    app = PendulumApp(root)
    root.mainloop()
