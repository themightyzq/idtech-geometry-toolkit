"""
idTech Map Generator - Progress Widget

This widget displays the current progress of map generation,
including phase indicators, progress bars, and status messages.
"""

from typing import Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QPushButton, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QTime
from PyQt5.QtGui import QFont, QPalette, QColor

from quake_levelgenerator.src.ui import style_constants as sc


class ProgressWidget(QWidget):
    """Widget for displaying generation progress."""
    
    # Signal emitted when cancel is requested
    cancel_requested = pyqtSignal()
    
    # Generation phases
    PHASE_LAYOUT = "Layout Generation"
    PHASE_3D = "3D Conversion"
    PHASE_COMPILE = "Map Compilation"
    PHASE_DEPLOY = "Deployment"
    
    def __init__(self, parent=None):
        """Initialize the progress widget."""
        super().__init__(parent)
        
        # State tracking
        self.current_phase = None
        self.start_time = None
        self.is_cancelled = False
        
        # Timer for elapsed time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_elapsed_time)
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the user interface."""
        # Set fixed height for the widget
        self.setFixedHeight(150)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 8, 16, 8)
        
        # Add separator line at the top
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)
        
        # Phase and status layout
        status_layout = QHBoxLayout()
        
        # Phase indicator
        phase_label = QLabel("Phase:")
        phase_label.setStyleSheet(f"font-weight: bold; color: {sc.TEXT_TERTIARY};")
        status_layout.addWidget(phase_label)

        self.phase_value = QLabel("Initializing...")
        self.phase_value.setStyleSheet(f"color: {sc.PRIMARY_ACTION}; font-weight: bold;")
        status_layout.addWidget(self.phase_value)
        
        status_layout.addStretch()
        
        # Elapsed time
        time_label = QLabel("Elapsed:")
        time_label.setStyleSheet(f"color: {sc.TEXT_TERTIARY};")
        status_layout.addWidget(time_label)

        self.elapsed_time = QLabel("00:00")
        self.elapsed_time.setStyleSheet(f"color: {sc.TEXT_PRIMARY};")
        status_layout.addWidget(self.elapsed_time)

        status_layout.addSpacing(20)

        # Estimated time
        eta_label = QLabel("ETA:")
        eta_label.setStyleSheet(f"color: {sc.TEXT_TERTIARY};")
        status_layout.addWidget(eta_label)

        self.eta_time = QLabel("Calculating...")
        self.eta_time.setStyleSheet(f"color: {sc.TEXT_PRIMARY};")
        status_layout.addWidget(self.eta_time)
        
        main_layout.addLayout(status_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(30)
        main_layout.addWidget(self.progress_bar)
        
        # Operation text and cancel button layout
        operation_layout = QHBoxLayout()
        
        # Current operation text
        self.operation_text = QLabel("Preparing generation...")
        self.operation_text.setStyleSheet(f"color: {sc.TEXT_SECONDARY}; font-style: italic;")
        operation_layout.addWidget(self.operation_text)
        
        operation_layout.addStretch()
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFixedWidth(80)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        operation_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(operation_layout)
        
        # Phase indicators
        phases_layout = QHBoxLayout()
        phases_layout.setSpacing(4)
        
        self.phase_indicators = {}
        phases = [self.PHASE_LAYOUT, self.PHASE_3D, self.PHASE_COMPILE, self.PHASE_DEPLOY]
        
        for i, phase in enumerate(phases):
            # Phase indicator widget
            indicator = QWidget()
            indicator.setFixedHeight(6)
            indicator.setStyleSheet(f"""
                QWidget {{
                    background-color: {sc.BG_LIGHT};
                    border-radius: {sc.BORDER_RADIUS_SM};
                }}
            """)
            self.phase_indicators[phase] = indicator
            phases_layout.addWidget(indicator)
            
            # Add separator between phases (except for last)
            if i < len(phases) - 1:
                phases_layout.addSpacing(2)
                
        main_layout.addLayout(phases_layout)
        
        # Apply widget styling
        self.setStyleSheet(f"""
            ProgressWidget {{
                background-color: {sc.BG_DARK};
                border-top: 1px solid {sc.BORDER_DARK};
            }}
            QProgressBar {{
                border: 1px solid {sc.BORDER_DARK};
                border-radius: {sc.BORDER_RADIUS_MD};
                background-color: {sc.BG_MEDIUM};
                text-align: center;
                color: {sc.TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {sc.PRIMARY_ACTION},
                    stop:1 {sc.PRIMARY_ACTION_HOVER}
                );
                border-radius: {sc.BORDER_RADIUS_SM};
            }}
            QPushButton {{
                background-color: {sc.DANGER_COLOR};
                color: white;
                border: none;
                border-radius: {sc.BORDER_RADIUS_MD};
                padding: {sc.SPACING_SM}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {sc.DANGER_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {sc.BG_PRESSED};
            }}
            QPushButton:disabled {{
                background-color: {sc.BORDER_LIGHT};
                color: {sc.TEXT_DISABLED};
            }}
        """)
        
    def start_generation(self):
        """Start the generation process."""
        self.is_cancelled = False
        self.current_phase = self.PHASE_LAYOUT
        self.start_time = QTime.currentTime()
        
        # Reset UI
        self.progress_bar.setValue(0)
        self.phase_value.setText(self.PHASE_LAYOUT)
        self.operation_text.setText("Initializing layout generation...")
        self.elapsed_time.setText("00:00")
        self.eta_time.setText("Calculating...")
        self.cancel_button.setEnabled(True)
        
        # Reset phase indicators
        for phase, indicator in self.phase_indicators.items():
            indicator.setStyleSheet(f"""
                QWidget {{
                    background-color: {sc.BG_LIGHT};
                    border-radius: {sc.BORDER_RADIUS_SM};
                }}
            """)
            
        # Highlight first phase
        self._highlight_phase(self.PHASE_LAYOUT)
        
        # Start elapsed time timer
        self.timer.start(1000)  # Update every second
        
    def update_progress(self, phase: str, progress: int, message: str):
        """Update the progress display."""
        if self.is_cancelled:
            return
            
        # Update phase if changed
        if phase != self.current_phase:
            self.current_phase = phase
            self.phase_value.setText(phase)
            self._highlight_phase(phase)
            
        # Update progress bar
        self.progress_bar.setValue(progress)
        
        # Update operation text
        self.operation_text.setText(message)
        
        # Update ETA (simple estimation)
        if progress > 0 and self.start_time:
            elapsed = self.start_time.secsTo(QTime.currentTime())
            if elapsed > 0:
                total_estimated = (elapsed * 100) / progress
                remaining = int(total_estimated - elapsed)
                if remaining > 0:
                    eta_mins = remaining // 60
                    eta_secs = remaining % 60
                    self.eta_time.setText(f"{eta_mins:02d}:{eta_secs:02d}")
                else:
                    self.eta_time.setText("Almost done...")
                    
        # Check if generation is complete
        if progress >= 100:
            self.generation_complete()
            
    def generation_complete(self):
        """Handle generation completion."""
        self.timer.stop()
        self.phase_value.setText("Complete")
        self.phase_value.setStyleSheet(f"color: {sc.PRIMARY_ACTION}; font-weight: bold;")
        self.operation_text.setText("Map generation completed successfully!")
        self.cancel_button.setEnabled(False)

        # Highlight all phases as complete
        for indicator in self.phase_indicators.values():
            indicator.setStyleSheet(f"""
                QWidget {{
                    background-color: {sc.PRIMARY_ACTION};
                    border-radius: {sc.BORDER_RADIUS_SM};
                }}
            """)
            
    def _highlight_phase(self, phase: str):
        """Highlight the current phase indicator."""
        phases = [self.PHASE_LAYOUT, self.PHASE_3D, self.PHASE_COMPILE, self.PHASE_DEPLOY]
        current_index = phases.index(phase) if phase in phases else -1

        for i, p in enumerate(phases):
            indicator = self.phase_indicators[p]

            if i < current_index:
                # Completed phase
                indicator.setStyleSheet(f"""
                    QWidget {{
                        background-color: {sc.PRIMARY_ACTION};
                        border-radius: {sc.BORDER_RADIUS_SM};
                    }}
                """)
            elif i == current_index:
                # Current phase (animated gradient)
                indicator.setStyleSheet(f"""
                    QWidget {{
                        background-color: qlineargradient(
                            x1:0, y1:0, x2:1, y2:0,
                            stop:0 {sc.PRIMARY_ACTION},
                            stop:0.5 #66BB6A,
                            stop:1 {sc.PRIMARY_ACTION}
                        );
                        border-radius: {sc.BORDER_RADIUS_SM};
                    }}
                """)
            else:
                # Pending phase
                indicator.setStyleSheet(f"""
                    QWidget {{
                        background-color: {sc.BG_LIGHT};
                        border-radius: {sc.BORDER_RADIUS_SM};
                    }}
                """)
                
    def _update_elapsed_time(self):
        """Update the elapsed time display."""
        if self.start_time:
            elapsed = self.start_time.secsTo(QTime.currentTime())
            mins = elapsed // 60
            secs = elapsed % 60
            self.elapsed_time.setText(f"{mins:02d}:{secs:02d}")
            
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.is_cancelled = True
        self.timer.stop()

        # Update UI
        self.phase_value.setText("Cancelled")
        self.phase_value.setStyleSheet(f"color: {sc.DANGER_COLOR}; font-weight: bold;")
        self.operation_text.setText("Generation cancelled by user.")
        self.cancel_button.setEnabled(False)
        
        # Emit cancel signal
        self.cancel_requested.emit()
        
    def reset(self):
        """Reset the progress widget to initial state."""
        self.timer.stop()
        self.is_cancelled = False
        self.current_phase = None
        self.start_time = None
        
        # Reset UI elements
        self.progress_bar.setValue(0)
        self.phase_value.setText("Ready")
        self.operation_text.setText("Waiting to start...")
        self.elapsed_time.setText("00:00")
        self.eta_time.setText("--:--")
        self.cancel_button.setEnabled(False)
        
        # Reset phase indicators
        for indicator in self.phase_indicators.values():
            indicator.setStyleSheet(f"""
                QWidget {{
                    background-color: {sc.BG_LIGHT};
                    border-radius: {sc.BORDER_RADIUS_SM};
                }}
            """)
            
    def set_indeterminate(self, message: str):
        """Set the progress bar to indeterminate mode."""
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.operation_text.setText(message)
        self.eta_time.setText("Unknown")
        
    def set_determinate(self):
        """Set the progress bar back to determinate mode."""
        self.progress_bar.setRange(0, 100)
        
    def show_error(self, error_message: str):
        """Display an error in the progress widget."""
        self.timer.stop()

        # Update UI for error state
        self.phase_value.setText("Error")
        self.phase_value.setStyleSheet(f"color: {sc.DANGER_COLOR}; font-weight: bold;")
        self.operation_text.setText(f"Error: {error_message}")
        self.cancel_button.setText("Close")
        self.cancel_button.setEnabled(True)

        # Set progress bar to error color
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {sc.DANGER_COLOR};
                border-radius: {sc.BORDER_RADIUS_MD};
                background-color: {sc.BG_MEDIUM};
                text-align: center;
                color: {sc.TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background-color: {sc.DANGER_COLOR};
                border-radius: {sc.BORDER_RADIUS_SM};
            }}
        """)