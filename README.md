# Production Line Optimization

This project is a Python application for optimizing production lines and performing sensitivity analysis. It provides a graphical user interface (GUI) for users to load factory and problem files, run optimization algorithms, and conduct sensitivity analysis on the results.

## Features

- Load factory and problem files
- Run optimization to find the best combination of production lines
- Perform sensitivity analysis on the optimized results
- User-friendly GUI built with customtkinter

## Requirements

The project requires the following Python libraries:

- pandas
- numpy
- itertools
- matplotlib
- pathlib
- tkinter
- customtkinter
- CTkMessagebox
- tkinter


## File Structure

The project consists of the following main files:

1. `PlhProjectSecondGUI.py`: The main application file containing the GUI and core functionality.
2. `Plhproject.py`: An alternative version of the project with similar functionality.
3. Various `.factory` and `.prob` files in the `Production_lines` directory, representing different factory configurations and production problems.

## How to Use

1. Run the `PlhProjectSecondGUI.py` file to start the application.
2. Use the GUI to select a factory file and a corresponding problem file.
3. Click "Run Optimization" to find the best combination of production lines.
4. After optimization, you can run a sensitivity analysis to examine how changes in costs affect the total production cost.

## Key Functions

- `open_factory(file_path)`: Loads factory data from a file.
- `open_prob(file_path)`: Loads problem data from a file.
- `optimum_combination(n, m, c, a, p)`: Calculates the optimal combination of production lines.
- `sensitivity_analysis(c, best_combination, a, p, selected_line, distribution, num_simulations)`: Performs sensitivity analysis on the optimized result.

## GUI Components

The application uses customtkinter to create a modern-looking GUI with the following main components:

- Dropdown menus for selecting factory and problem files
- Buttons for running optimization and sensitivity analysis
- A results display area

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.