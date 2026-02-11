from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from ipywidgets import widgets
from IPython.display import display


class ValidationResult:
    """
    Container for validation results with visualization capabilities.
    """

    def __init__(self, results: Dict[str, Dict]):
        """
        Parameters
        ----------
        results : Dict[str, Dict]
            Dictionary containing validation data for each dataset.
            Structure:
            {
                'dataset_name': {
                    'metrics': {'RMSE': float, ...},
                    'y_pred': pd.DataFrame,
                    'y_true': pd.DataFrame,
                    'x_val': pd.DataFrame
                }
            }
        """
        self.results = results

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame of metrics for all datasets.
        """
        data = []
        for name, res in self.results.items():
            row = {'Dataset': name}
            row.update(res['metrics'])
            data.append(row)
        return pd.DataFrame(data).set_index('Dataset')

    def _is_scalar(self, dataset_name: str) -> bool:
        """
        Check if a dataset contains scalar values (single point per sample).

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to check.

        Returns
        -------
        bool
            True if the dataset has only one column (scalar), False otherwise (curve).
        """
        return self.results[dataset_name]['y_true'].shape[1] == 1

    def dashboard(self):
        """
        Displays an interactive Jupyter dashboard.

        Features:
        - Dataset selection dropdown.
        - Parity plot (Predicted vs Actual).
        - Curve comparison plot (Predicted vs Actual).
        - Click on Parity Plot points to update the Curve Plot.
        """
        dataset_names = list(self.results.keys())
        if not dataset_names:
            print("No validation results to display.")
            return

        # Widgets
        ds_dropdown: widgets.Dropdown = widgets.Dropdown(
            options=dataset_names,
            value=dataset_names[0],
            description='Dataset:'
        )

        # Figures
        parity_fig: go.FigureWidget = go.FigureWidget()
        curve_fig: go.FigureWidget = go.FigureWidget()

        # Layouts
        parity_fig.update_layout(
            title="Parity Plot (Click to inspect)",
            xaxis_title="Actual Value",
            yaxis_title="Predicted Value",
            showlegend=False,
            width=500,
            height=500,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        )
        # Add 45-degree line (Initial placeholder)
        parity_fig.add_shape(
            type="line", line=dict(color="gray", width=2), opacity=0.3,
            layer="below",
            x0=0, y0=0, x1=1, y1=1
        )

        curve_fig.update_layout(
            title="Curve Comparison",
            xaxis_title="Curve Point",
            yaxis_title="Value",
            width=500,
            height=500
        )

        # Container for curve_fig to control visibility
        curve_container = widgets.Box([curve_fig], layout=widgets.Layout(display='block'))

        def update_charts(dataset_name, sample_idx=0):
            is_scalar = self._is_scalar(dataset_name)
            res = self.results[dataset_name]
            y_pred = res['y_pred']
            y_true = res['y_true']
            x_val = res['x_val']

            # 1. Update Parity Plot
            # Flatten data for parity plot
            y_true_flat = y_true.values.flatten()
            y_pred_flat = y_pred.values.flatten()

            # Create sample indices for mapping back
            n_samples, n_points = y_true.shape
            # Positional indices for coloring (0, 1, 2, ...)
            color_indices = np.repeat(np.arange(n_samples), n_points)
            # Actual x_val index values for display in hover
            sample_labels = np.repeat(x_val.index.values, n_points)

            # Update 45-degree line range
            min_val = min(y_true_flat.min(), y_pred_flat.min())
            max_val = max(y_true_flat.max(), y_pred_flat.max())

            # Add padding
            padding = (max_val - min_val) * 0.05
            min_val -= padding
            max_val += padding

            # Generate colors for hover
            if n_samples > 1:
                unique_norm = np.linspace(0, 1, n_samples)
            else:
                unique_norm = [0.0]

            unique_colors = sample_colorscale('Turbo', unique_norm)
            hover_colors = np.repeat(unique_colors, n_points)

            with parity_fig.batch_update():
                parity_fig.data = []
                parity_fig.add_trace(go.Scattergl(
                    x=y_true_flat,
                    y=y_pred_flat,
                    mode='markers',
                    marker=dict(
                        opacity=0.7,
                        size=8,
                        color=color_indices,
                        colorscale='Turbo',
                        showscale=False
                    ),
                    hoverlabel=dict(bgcolor=hover_colors),
                    customdata=sample_labels,
                    hovertemplate="Actual: %{x}<br>Pred: %{y}<br>Sample: %{customdata}<extra></extra>"
                ))

                parity_fig.update_layout(shapes=[dict(
                    type="line", line=dict(color="gray", width=2), opacity=0.3,
                    layer="below",
                    x0=min_val, y0=min_val, x1=max_val, y1=max_val
                )])

            # 2. Update Curve Plot (only for non-scalar datasets)
            if is_scalar:
                curve_container.layout.display = 'none'
            else:
                curve_container.layout.display = 'block'
                # Use actual index value for initial curve plot
                initial_label = x_val.index[sample_idx]
                update_curve_plot(dataset_name, initial_label)

            # Update Parity Plot Title with Global SRMSE
            srmse_pct = res['metrics'].get('SRMSE', 0) * 100
            parity_fig.update_layout(title=f"Parity Plot (Global SRMSE: {srmse_pct:.2f}%)")

            # Re-attach click callback to the new trace (only useful for curve datasets)
            if not is_scalar:
                parity_fig.data[0].on_click(on_parity_click)

        def update_curve_plot(dataset_name, sample_label):
            res = self.results[dataset_name]
            y_pred: pd.DataFrame = res['y_pred']
            y_true: pd.DataFrame = res['y_true']
            x_val = res['x_val']
            # Convert sample label (actual index) to positional index for .iloc[]
            sample_pos = x_val.index.get_loc(sample_label)
            y_pred_sample: pd.Series = y_pred.iloc[sample_pos]
            y_true_sample: pd.Series = y_true.iloc[sample_pos]

            # Calculate Curve-specific SRMSE
            rmse_curve = np.sqrt(np.mean((y_pred_sample.values - y_true_sample.values) ** 2))
            std_curve = np.std(y_true_sample)
            if std_curve == 0:
                srmse_curve = 0.0 if rmse_curve == 0 else np.inf
            else:
                srmse_curve = rmse_curve / std_curve

            srmse_curve_pct = srmse_curve * 100

            with curve_fig.batch_update():
                curve_fig.data = []
                curve_fig.add_trace(go.Scatter(
                    y=y_true_sample, name="Actual", line=dict(color='black')
                ))
                curve_fig.add_trace(go.Scatter(
                    y=y_pred_sample, name="Predicted", line=dict(color='blue', dash='dash')
                ))
                curve_fig.update_layout(title=f"Sample {sample_label} (SRMSE: {srmse_curve_pct:.2f}%)")

        # Callbacks
        def on_dataset_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                update_charts(change['new'])

        def on_parity_click(trace, points, selector):
            # Skip for scalar datasets - no curve to display
            if self._is_scalar(ds_dropdown.value):
                return
            if points.point_inds:
                # Get the sample label (actual x_val index) from customdata
                # points.point_inds[0] gives the index in the flattened array
                # We stored actual index values (sample_labels) in customdata
                flat_idx = points.point_inds[0]
                sample_label = trace.customdata[flat_idx]
                update_curve_plot(ds_dropdown.value, sample_label)

        ds_dropdown.observe(on_dataset_change)

        # Initialize
        update_charts(ds_dropdown.value)

        # Display
        ui = widgets.VBox([
            ds_dropdown,
            widgets.HBox([parity_fig, curve_container],
                         layout=widgets.Layout(justify_content='center'))
        ])
        display(ui)

    def save_html(self, filename: str):
        """
        Exports the validation results to an HTML file.

        Features:
        - Responsive layout (Flexbox).
        - Dropdown to select Dataset.
        - Parity plot and Curve plot side-by-side (or stacked on mobile).
        """
        import plotly.io as pio

        dataset_names = list(self.results.keys())
        if not dataset_names:
            return

        # Track which datasets are scalar vs curve
        is_scalar_list = [self._is_scalar(ds_name) for ds_name in dataset_names]

        # 1. Create Parity Figure
        fig_parity = go.Figure()
        fig_parity.update_layout(
            title="Parity Plot",
            xaxis_title="Actual Value",
            yaxis_title="Predicted Value",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        for i, ds_name in enumerate(dataset_names):
            res = self.results[ds_name]
            y_true_flat = res['y_true'].values.flatten()
            y_pred_flat = res['y_pred'].values.flatten()
            x_val = res['x_val']

            # Scatter Trace
            # Create sample indices for coloring
            n_samples, n_points = res['y_true'].shape
            # Positional indices for coloring
            color_indices = np.repeat(np.arange(n_samples), n_points)
            # Actual x_val index values for display
            sample_labels = np.repeat(x_val.index.values, n_points)

            # Generate colors for hover
            if n_samples > 1:
                unique_norm = np.linspace(0, 1, n_samples)
            else:
                unique_norm = [0.0]

            unique_colors = sample_colorscale('Turbo', unique_norm)
            hover_colors = np.repeat(unique_colors, n_points)

            fig_parity.add_trace(go.Scattergl(
                x=y_true_flat, y=y_pred_flat,
                mode='markers',
                name=f"{ds_name}",
                visible=(i == 0),
                marker=dict(
                    opacity=0.7,
                    size=8,
                    color=color_indices,
                    colorscale='Turbo',
                    showscale=False
                ),
                hoverlabel=dict(bgcolor=hover_colors),
                customdata=sample_labels,
                hovertemplate="Actual: %{x}<br>Pred: %{y}<br>Sample: %{customdata}<extra></extra>",
                showlegend=False
            ))

            # 45-degree Line Trace (using trace instead of shape for visibility control)
            min_val = min(y_true_flat.min(), y_pred_flat.min())
            max_val = max(y_true_flat.max(), y_pred_flat.max())
            padding = (max_val - min_val) * 0.05
            min_val -= padding
            max_val += padding

            fig_parity.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                line=dict(color="gray", width=2),
                opacity=0.3,
                showlegend=False,
                visible=(i == 0),
                hoverinfo='skip'
            ))

        # 2. Create Curve Figure
        fig_curve = go.Figure()
        # Get initial sample label from first dataset
        first_ds = self.results[dataset_names[0]]
        initial_sample_label = first_ds['x_val'].index[0]
        fig_curve.update_layout(
            title=f"Curve Comparison (Sample {initial_sample_label})",
            xaxis_title="Curve Point",
            yaxis_title="Value",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        for i, ds_name in enumerate(dataset_names):
            res = self.results[ds_name]
            # Only showing first sample for HTML export simplicity
            y_true = res['y_true'].iloc[0]
            y_pred = res['y_pred'].iloc[0]

            fig_curve.add_trace(go.Scatter(
                y=y_true, mode='lines', line=dict(color='black'),
                name='Actual', visible=(i == 0), showlegend=True
            ))
            fig_curve.add_trace(go.Scatter(
                y=y_pred, mode='lines', line=dict(color='blue', dash='dash'),
                name='Predicted', visible=(i == 0), showlegend=True
            ))

        # 3. Generate HTML
        div_parity = pio.to_html(fig_parity, full_html=False, include_plotlyjs='cdn', div_id='parity_plot')
        div_curve = pio.to_html(fig_curve, full_html=False, include_plotlyjs=False, div_id='curve_plot')

        # Dropdown Options
        options_html = ""
        for i, ds_name in enumerate(dataset_names):
            options_html += f'<option value="{i}">{ds_name}</option>'

        # Generate JavaScript array for scalar dataset flags
        is_scalar_js = str(is_scalar_list).lower()  # Convert Python list to JS array format

        # Determine initial curve plot visibility
        initial_curve_display = 'none' if is_scalar_list[0] else 'block'

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Validation Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
                .controls {{ margin-bottom: 20px; }}
                .container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: center;
                }}
                .plot-div {{
                    flex: 1 1 500px;
                    min-width: 300px;
                    max-width: 800px;
                    height: 500px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                }}
            </style>
        </head>
        <body>
            <h1>Validation Report</h1>
            <div class="controls">
                <label for="dataset-select">Select Dataset:</label>
                <select id="dataset-select" onchange="updatePlots(this.value)">
                    {options_html}
                </select>
            </div>
            <div class="container">
                <div class="plot-div">{div_parity}</div>
                <div class="plot-div" id="curve_container" style="display: {initial_curve_display};">{div_curve}</div>
            </div>
            <script>
                var isScalar = {is_scalar_js};
                
                function updatePlots(idx) {{
                    idx = parseInt(idx);
                    var n_datasets = {len(dataset_names)};
                    
                    // Parity: 2 traces per dataset (Scatter + Line)
                    var vis_parity = new Array(n_datasets * 2).fill(false);
                    vis_parity[idx * 2] = true;
                    vis_parity[idx * 2 + 1] = true;
                    
                    // Curve: 2 traces per dataset (Actual + Pred)
                    var vis_curve = new Array(n_datasets * 2).fill(false);
                    vis_curve[idx * 2] = true;
                    vis_curve[idx * 2 + 1] = true;

                    Plotly.restyle('parity_plot', {{visible: vis_parity}});
                    Plotly.restyle('curve_plot', {{visible: vis_curve}});
                    
                    // Toggle curve container visibility based on dataset type
                    var curveContainer = document.getElementById('curve_container');
                    curveContainer.style.display = isScalar[idx] ? 'none' : 'block';
                }}
            </script>
        </body>
        </html>
        """

        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)


class LearningCurveResult:
    """
    Container for learning curve results with visualization capabilities.

    Stores validation metrics for each training size, enabling analysis of
    how model performance improves with increasing training data.
    """

    def __init__(self, results: Dict[str, Dict[int, Dict]]):
        """
        Parameters
        ----------
        results : Dict[str, Dict[int, Dict]]
            Dictionary containing learning curve data for each dataset.
            Structure:
            {
                'dataset_name': {
                    train_size: {
                        'metrics': {'RMSE': float, 'MAE': float, ...},
                        'y_pred': pd.DataFrame (optional),
                        'y_true': pd.DataFrame (optional),
                        'x_val': pd.DataFrame
                    },
                    ...
                }
            }
        """
        self.results = results

    @property
    def dataset_names(self) -> List[str]:
        """Returns list of dataset names."""
        return list(self.results.keys())

    @property
    def train_sizes(self) -> List[int]:
        """Returns list of training sizes (from first dataset)."""
        if not self.results:
            return []
        first_ds = next(iter(self.results.values()))
        return sorted(first_ds.keys())

    @property
    def metric_names(self) -> List[str]:
        """Returns list of metric names."""
        if not self.results:
            return []
        first_ds = next(iter(self.results.values()))
        if not first_ds:
            return []
        first_size = next(iter(first_ds.values()))
        return list(first_size['metrics'].keys())

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame of metrics for all datasets and training sizes.

        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex (Dataset, TrainSize) and metric columns.
        """
        data = []
        for ds_name, sizes_dict in self.results.items():
            for train_size, res in sizes_dict.items():
                row = {'Dataset': ds_name, 'TrainSize': train_size}
                row.update(res['metrics'])
                data.append(row)
        return pd.DataFrame(data).set_index(['Dataset', 'TrainSize'])

    def plot(self, metric: str = 'SRMSE') -> go.Figure:
        """
        Creates a learning curve plot for the specified metric.

        Parameters
        ----------
        metric : str, optional
            The metric to plot. Default is 'SRMSE'.

        Returns
        -------
        go.Figure
            Plotly figure with training size on X-axis and metric on Y-axis.
        """
        fig = go.Figure()

        colors = sample_colorscale(
            'Turbo',
            [i / max(1, len(self.dataset_names) - 1) for i in range(len(self.dataset_names))]
        )

        for i, ds_name in enumerate(self.dataset_names):
            sizes = []
            values = []
            for train_size in self.train_sizes:
                if train_size in self.results[ds_name]:
                    sizes.append(train_size)
                    values.append(self.results[ds_name][train_size]['metrics'][metric])

            fig.add_trace(go.Scatter(
                x=sizes,
                y=values,
                mode='lines+markers',
                name=ds_name,
                line=dict(color=colors[i]),
                marker=dict(size=8)
            ))

        fig.update_layout(
            title=f'Learning Curve: {metric}',
            xaxis_title='Training Size',
            yaxis_title=metric,
            yaxis=dict(exponentformat='e'),
            hovermode='x unified',
            width=700,
            height=500
        )

        return fig

    def dashboard(self):
        """
        Displays an interactive Jupyter dashboard.

        Features:
        - Dataset selection dropdown (multi-select).
        - Metric selection dropdown.
        - Learning curve plot that updates on selection changes.
        """
        if not self.dataset_names:
            print('No learning curve results to display.')
            return

        # Widgets
        ds_select: widgets.SelectMultiple = widgets.SelectMultiple(
            options=self.dataset_names,
            value=tuple(self.dataset_names),
            description='Datasets:',
            rows=min(5, len(self.dataset_names))
        )

        metric_dropdown: widgets.Dropdown = widgets.Dropdown(
            options=self.metric_names,
            value='SRMSE' if 'SRMSE' in self.metric_names else self.metric_names[0],
            description='Metric:'
        )

        # Figure
        fig: go.FigureWidget = go.FigureWidget()
        fig.update_layout(
            title='Learning Curve',
            xaxis_title='Training Size',
            yaxis_title='Metric',
            hovermode='x unified',
            width=700,
            height=500
        )

        def update_plot(*args):
            selected_datasets = list(ds_select.value)
            metric = metric_dropdown.value

            colors = sample_colorscale(
                'Turbo',
                [i / max(1, len(selected_datasets) - 1) for i in range(len(selected_datasets))]
            )

            with fig.batch_update():
                fig.data = []
                for i, ds_name in enumerate(selected_datasets):
                    sizes = []
                    values = []
                    for train_size in self.train_sizes:
                        if train_size in self.results[ds_name]:
                            sizes.append(train_size)
                            values.append(self.results[ds_name][train_size]['metrics'][metric])

                    fig.add_trace(go.Scatter(
                        x=sizes,
                        y=values,
                        mode='lines+markers',
                        name=ds_name,
                        line=dict(color=colors[i] if len(selected_datasets) > 1 else 'blue'),
                        marker=dict(size=8)
                    ))

                fig.update_layout(
                    title=f"Learning Curve: {metric}",
                    yaxis_title=metric
                )

        # Callbacks
        ds_select.observe(update_plot, names='value')
        metric_dropdown.observe(update_plot, names='value')

        # Initialize
        update_plot()

        # Layout
        controls = widgets.VBox([ds_select, metric_dropdown])
        ui = widgets.HBox([controls, fig], layout=widgets.Layout(justify_content='center'))
        display(ui)

    def save_html(self, filename: str):
        """
        Exports the learning curve results to an HTML file.

        Parameters
        ----------
        filename : str
            Path to the output HTML file.
        """
        import plotly.io as pio

        if not self.dataset_names:
            return

        # Create figure with all datasets and metrics
        fig = go.Figure()

        colors = sample_colorscale(
            'Turbo',
            [i / max(1, len(self.dataset_names) - 1) for i in range(len(self.dataset_names))]
        )

        # Add traces for each dataset and metric combination
        # Structure: for each metric, for each dataset -> one trace
        for metric_idx, metric in enumerate(self.metric_names):
            for ds_idx, ds_name in enumerate(self.dataset_names):
                sizes = []
                values = []
                for train_size in self.train_sizes:
                    if train_size in self.results[ds_name]:
                        sizes.append(train_size)
                        values.append(self.results[ds_name][train_size]['metrics'][metric])

                fig.add_trace(go.Scatter(
                    x=sizes,
                    y=values,
                    mode='lines+markers',
                    name=ds_name,
                    line=dict(color=colors[ds_idx]),
                    marker=dict(size=8),
                    visible=(metric_idx == 0),  # Only first metric visible initially
                    showlegend=(metric_idx == 0)
                ))

        fig.update_layout(
            title=f'Learning Curve: {self.metric_names[0]}',
            xaxis_title='Training Size',
            yaxis_title=self.metric_names[0],
            yaxis=dict(exponentformat='e'),
            hovermode='x unified',
            margin=dict(l=20, r=20, t=60, b=20)
        )

        # Generate HTML
        div_plot = pio.to_html(fig, full_html=False, include_plotlyjs='cdn', div_id='learning_curve_plot')

        # Metric dropdown options
        metric_options_html = ""
        for i, metric in enumerate(self.metric_names):
            metric_options_html += f'<option value="{i}">{metric}</option>'

        n_datasets = len(self.dataset_names)
        n_metrics = len(self.metric_names)
        metric_names_js = str(self.metric_names)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Learning Curve Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
                .controls {{ margin-bottom: 20px; }}
                .controls select {{ margin-right: 20px; }}
                .plot-container {{
                    display: flex;
                    justify-content: center;
                }}
                .plot-div {{
                    width: 100%;
                    max-width: 900px;
                    height: 500px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                }}
            </style>
        </head>
        <body>
            <h1>Learning Curve Report</h1>
            <div class="controls">
                <label for="metric-select">Select Metric:</label>
                <select id="metric-select" onchange="updatePlot(this.value)">
                    {metric_options_html}
                </select>
            </div>
            <div class="plot-container">
                <div class="plot-div">{div_plot}</div>
            </div>
            <script>
                var metricNames = {metric_names_js};
                var nDatasets = {n_datasets};
                var nMetrics = {n_metrics};
                
                function updatePlot(metricIdx) {{
                    metricIdx = parseInt(metricIdx);
                    
                    // Each metric has nDatasets traces
                    var totalTraces = nDatasets * nMetrics;
                    var visibility = new Array(totalTraces).fill(false);
                    var showLegend = new Array(totalTraces).fill(false);
                    
                    // Show traces for selected metric
                    for (var i = 0; i < nDatasets; i++) {{
                        var traceIdx = metricIdx * nDatasets + i;
                        visibility[traceIdx] = true;
                        showLegend[traceIdx] = true;
                    }}
                    
                    Plotly.restyle('learning_curve_plot', {{
                        visible: visibility,
                        showlegend: showLegend
                    }});
                    
                    Plotly.relayout('learning_curve_plot', {{
                        title: 'Learning Curve: ' + metricNames[metricIdx],
                        'yaxis.title': metricNames[metricIdx]
                    }});
                }}
            </script>
        </body>
        </html>
        """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)


class ConfigSearchResult:
    """
    Container for configuration search results with visualization capabilities.

    Stores validation metrics for each Kriging configuration tested, enabling
    analysis of which hyperparameter combinations perform best.
    """

    def __init__(self, results: Dict[str, Dict[int, Dict]], failures: List[Dict] = None):
        """
        Parameters
        ----------
        results : Dict[str, Dict[int, Dict]]
            Dictionary containing config search data for each dataset.
            Structure:
            {
                'dataset_name': {
                    config_id: {
                        'metrics': {'RMSE': float, 'MAE': float, ...},
                        'config': KrigingConfig,
                        'y_pred': pd.DataFrame (optional),
                        'y_true': pd.DataFrame (optional),
                        'x_val': pd.DataFrame
                    },
                    ...
                }
            }
        failures : List[Dict], optional
            List of failed configurations. Each entry is a dict with:
            {
                'config': KrigingConfig,
                'error': str (error message)
            }
        """
        self.results = results
        self.failures = failures or []

    @property
    def n_successful(self) -> int:
        """Returns the number of successful configurations."""
        if not self.results:
            return 0
        first_ds = next(iter(self.results.values()))
        return len(first_ds)

    @property
    def n_failed(self) -> int:
        """Returns the number of failed configurations."""
        return len(self.failures)

    @property
    def dataset_names(self) -> List[str]:
        """Returns list of dataset names."""
        return list(self.results.keys())

    @property
    def metric_names(self) -> List[str]:
        """Returns list of metric names."""
        if not self.results:
            return []
        first_ds = next(iter(self.results.values()))
        if not first_ds:
            return []
        first_config = next(iter(first_ds.values()))
        return list(first_config['metrics'].keys())

    @property
    def configs(self) -> List:
        """Returns list of all successfully tested KrigingConfig objects."""
        if not self.results:
            return []
        first_ds = next(iter(self.results.values()))
        return [res['config'] for res in first_ds.values()]

    def failures_summary(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame of failed configurations.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for config parameters and error message.
            Empty DataFrame if no failures.
        """
        if not self.failures:
            return pd.DataFrame(columns=[
                'basis_function', 'stationarity', 'pulsation', 'nugget_effect', 'error'
            ])

        data = []
        for failure in self.failures:
            config = failure['config']
            data.append({
                'basis_function': config.basis_function,
                'stationarity': config.stationarity,
                'pulsation': config.pulsation,
                'nugget_effect': config.nugget_effect,
                'error': failure['error']
            })
        return pd.DataFrame(data)

    def summary(self, include_failures: bool = False) -> pd.DataFrame:
        """
        Returns a summary DataFrame of metrics for all datasets and configurations.

        Parameters
        ----------
        include_failures : bool, optional
            If True, includes failed configurations with NaN metrics and an
            'error' column containing the failure message. Default is False.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for config parameters and all metrics.
            Index is 'Dataset'. If include_failures=True, adds 'error' column.
        """
        data = []
        for ds_name, configs_dict in self.results.items():
            for config_id, res in configs_dict.items():
                config = res['config']
                row = {
                    'Dataset': ds_name,
                    'basis_function': config.basis_function,
                    'stationarity': config.stationarity,
                    'pulsation': config.pulsation,
                    'nugget_effect': config.nugget_effect
                }
                row.update(res['metrics'])
                if include_failures:
                    row['error'] = None
                data.append(row)

        if include_failures:
            for failure in self.failures:
                config = failure['config']
                row = {
                    'Dataset': None,
                    'basis_function': config.basis_function,
                    'stationarity': config.stationarity,
                    'pulsation': config.pulsation,
                    'nugget_effect': config.nugget_effect,
                    'error': failure['error']
                }
                # Add NaN for all metrics
                for metric in self.metric_names:
                    row[metric] = np.nan
                data.append(row)

        return pd.DataFrame(data).set_index('Dataset')

    def best(self, weights: Dict[str, float] = None) -> Dict:
        """
        Returns the best configuration for each dataset based on weighted metrics.

        Parameters
        ----------
        weights : Dict[str, float], optional
            Weights for each metric. Higher weight = more importance.
            Default is {'SRMSE': 1.0} (optimize for SRMSE only).
            Example: {'SRMSE': 1.0, 'MAE': 0.5} weights SRMSE twice as much as MAE.

        Returns
        -------
        Dict[str, KrigingConfig]
            Dictionary mapping dataset name to the optimal KrigingConfig.

        Notes
        -----
        All metrics are treated as "lower is better". The weighted score is
        computed as: score = sum(metric_value * weight) / sum(weights)
        """
        if weights is None:
            weights = {'SRMSE': 1.0}

        best_configs = {}
        for ds_name, configs_dict in self.results.items():
            best_score = float('inf')
            best_config = None

            for config_id, res in configs_dict.items():
                metrics = res['metrics']
                # Compute weighted score (lower is better)
                total_weight = sum(weights.get(m, 0) for m in metrics)
                if total_weight == 0:
                    continue

                weighted_sum = sum(
                    metrics[m] * weights.get(m, 0)
                    for m in metrics
                )
                score = weighted_sum / total_weight

                if score < best_score:
                    best_score = score
                    best_config = res['config']

            best_configs[ds_name] = best_config

        return best_configs

    def plot(self, metric: str = 'SRMSE', max_configs: int = 20) -> go.Figure:
        """
        Creates a bar chart comparing configurations for a specified metric.

        Parameters
        ----------
        metric : str, optional
            The metric to plot. Default is 'SRMSE'.
        max_configs : int, optional
            Maximum number of configurations to display (sorted by metric).
            Default is 20.

        Returns
        -------
        go.Figure
            Plotly figure with configurations on X-axis and metric on Y-axis.
        """
        fig = go.Figure()

        colors = sample_colorscale(
            'Turbo',
            [i / max(1, len(self.dataset_names) - 1) for i in range(len(self.dataset_names))]
        )

        for i, ds_name in enumerate(self.dataset_names):
            configs_dict = self.results[ds_name]

            # Sort by metric and limit
            sorted_configs = sorted(
                configs_dict.items(),
                key=lambda x: x[1]['metrics'][metric]
            )[:max_configs]

            labels = []
            values = []
            for config_id, res in sorted_configs:
                cfg = res['config']
                label = f'bf={cfg.basis_function}, st={cfg.stationarity}, nug={cfg.nugget_effect:.1f}'
                labels.append(label)
                values.append(res['metrics'][metric])

            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                name=ds_name,
                marker_color=colors[i] if len(self.dataset_names) > 1 else 'steelblue'
            ))

        fig.update_layout(
            title=f'Configuration Comparison: {metric}',
            xaxis_title='Configuration',
            yaxis_title=metric,
            yaxis=dict(exponentformat='e', type='log'),
            barmode='group',
            xaxis_tickangle=-45,
            width=900,
            height=500
        )

        return fig

    def dashboard(self):
        """
        Displays an interactive Jupyter dashboard with weight sliders.

        Features:
        - Dataset selection dropdown.
        - Weight sliders for each metric (0-1 range).
        - Bar chart of configurations ranked by weighted score.
        - Best configuration highlighted.
        - Reset Weights button.
        """
        if not self.dataset_names:
            print('No config search results to display.')
            return

        # Widgets
        ds_dropdown: widgets.Dropdown = widgets.Dropdown(
            options=self.dataset_names,
            value=self.dataset_names[0],
            description='Dataset:'
        )

        # Weight sliders for each metric
        weight_sliders = {}
        for metric in self.metric_names:
            default_val = 1.0 if metric == 'SRMSE' else 0.0
            weight_sliders[metric] = widgets.FloatSlider(
                value=default_val,
                min=0.0,
                max=1.0,
                step=0.05,
                description=f'{metric}:',
                continuous_update=False,
                style={'description_width': '80px'},
                layout=widgets.Layout(width='280px'),
                readout_format='.2f'
            )

        # Reset button
        reset_btn = widgets.Button(
            description='Reset Weights',
            button_style='warning',
            layout=widgets.Layout(width='120px')
        )

        # Figure
        fig: go.FigureWidget = go.FigureWidget()
        fig.update_layout(
            title='Configuration Comparison (Weighted Score)',
            xaxis_title='Configuration',
            yaxis_title='Weighted Score',
            yaxis=dict(exponentformat='e', type='log'),
            xaxis_tickangle=-45,
            width=700,
            height=500
        )

        def get_weights():
            return {m: weight_sliders[m].value for m in self.metric_names}

        def update_plot(*args):
            ds_name = ds_dropdown.value
            weights = get_weights()
            configs_dict = self.results[ds_name]

            # Calculate weighted scores
            scores = []
            for config_id, res in configs_dict.items():
                metrics = res['metrics']
                total_weight = sum(weights.get(m, 0) for m in metrics)
                if total_weight == 0:
                    score = 0
                else:
                    weighted_sum = sum(metrics[m] * weights.get(m, 0) for m in metrics)
                    score = weighted_sum / total_weight

                cfg = res['config']
                label = f'bf={cfg.basis_function}, st={cfg.stationarity}, nug={cfg.nugget_effect:.1f}'
                scores.append((label, score, config_id))

            # Sort by score (lower is better)
            scores.sort(key=lambda x: x[1])

            # Limit to top 20
            scores = scores[:20]

            labels = [s[0] for s in scores]
            values = [s[1] for s in scores]

            # Highlight best (first after sort)
            colors = ['gold' if i == 0 else 'steelblue' for i in range(len(scores))]

            with fig.batch_update():
                fig.data = []
                fig.add_trace(go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    showlegend=False
                ))

                # Update title with best config info
                if scores:
                    best_label = scores[0][0]
                    best_score = scores[0][1]
                    fig.update_layout(
                        title=f'Best: {best_label} (score: {best_score:.4e})'
                    )

        def reset_weights(btn):
            for metric, slider in weight_sliders.items():
                slider.value = 1.0 if metric == 'SRMSE' else 0.0

        # Connect callbacks
        ds_dropdown.observe(update_plot, names='value')
        for slider in weight_sliders.values():
            slider.observe(update_plot, names='value')
        reset_btn.on_click(reset_weights)

        # Initialize
        update_plot()

        # Layout
        slider_box = widgets.VBox(
            [widgets.Label('Metric Weights:')] +
            list(weight_sliders.values()) +
            [reset_btn]
        )
        controls = widgets.VBox([ds_dropdown, slider_box])

        ui = widgets.HBox([controls, fig], layout=widgets.Layout(justify_content='center'))
        display(ui)

    def save_html(self, filename: str):
        """
        Exports the configuration search results to an HTML file with interactive sliders.

        Parameters
        ----------
        filename : str
            Path to the output HTML file.
        """
        import json
        import plotly.io as pio

        if not self.dataset_names:
            return

        # Prepare data for JavaScript
        js_data = {}
        for ds_name, configs_dict in self.results.items():
            js_data[ds_name] = []
            for config_id, res in configs_dict.items():
                cfg = res['config']
                js_data[ds_name].append({
                    'label': f'bf={cfg.basis_function}, st={cfg.stationarity}, nug={cfg.nugget_effect:.1f}',
                    'metrics': res['metrics']
                })

        data_json = json.dumps(js_data)
        metric_names_json = json.dumps(self.metric_names)
        dataset_names_json = json.dumps(self.dataset_names)

        # Initial figure (will be updated by JS)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[], y=[], marker_color=[]))
        fig.update_layout(
            title='Configuration Comparison (Weighted Score)',
            xaxis_title='Configuration',
            yaxis_title='Weighted Score',
            yaxis=dict(exponentformat='e'),
            xaxis_tickangle=-45,
            margin=dict(l=20, r=20, t=60, b=120)
        )

        div_plot = pio.to_html(fig, full_html=False, include_plotlyjs='cdn', div_id='config_plot')

        # Dataset dropdown options
        ds_options_html = '\n'.join(
            f'<option value="{ds}">{ds}</option>'
            for ds in self.dataset_names
        )

        # Slider HTML for each metric
        slider_html = '\n'.join(
            f'''<div class="slider-row">
                <label for="{metric.lower().replace(' ', '_')}-weight">{metric}:</label>
                <input type="range" id="{metric.lower().replace(' ', '_')}-weight"
                       min="0" max="1" step="0.05" value="{'1.0' if metric == 'SRMSE' else '0.0'}"
                       oninput="updateWeights()">
                <span id="{metric.lower().replace(' ', '_')}-value">{'1.00' if metric == 'SRMSE' else '0.00'}</span>
            </div>'''
            for metric in self.metric_names
        )

        # JavaScript for weight reading
        weight_readers_js = ',\n'.join(
            f"'{metric}': parseFloat(document.getElementById('{metric.lower().replace(' ', '_')}-weight').value)"
            for metric in self.metric_names
        )

        value_updaters_js = '\n'.join(
            f"document.getElementById('{metric.lower().replace(' ', '_')}-value').innerText = "
            f"document.getElementById('{metric.lower().replace(' ', '_')}-weight').value;"
            for metric in self.metric_names
        )

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Configuration Search Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
                .container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: center;
                }}
                .controls {{
                    flex: 0 0 300px;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background: #f9f9f9;
                }}
                .controls h3 {{ margin-top: 0; }}
                .controls select {{
                    width: 100%;
                    padding: 8px;
                    margin-bottom: 15px;
                    font-size: 14px;
                }}
                .slider-row {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin: 8px 0;
                }}
                .slider-row label {{
                    min-width: 80px;
                    font-weight: bold;
                }}
                .slider-row input[type="range"] {{
                    flex: 1;
                }}
                .slider-row span {{
                    min-width: 40px;
                    text-align: right;
                }}
                .reset-btn {{
                    width: 100%;
                    padding: 10px;
                    margin-top: 15px;
                    background: #f0ad4e;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                }}
                .reset-btn:hover {{ background: #ec971f; }}
                .plot-div {{
                    flex: 1 1 600px;
                    min-width: 400px;
                    max-width: 900px;
                    height: 500px;
                }}
            </style>
        </head>
        <body>
            <h1>Configuration Search Report</h1>
            <div class="container">
                <div class="controls">
                    <h3>Settings</h3>
                    <label for="dataset-select">Dataset:</label>
                    <select id="dataset-select" onchange="updateWeights()">
                        {ds_options_html}
                    </select>
                    <h4>Metric Weights</h4>
                    {slider_html}
                    <button class="reset-btn" onclick="resetWeights()">Reset Weights</button>
                </div>
                <div class="plot-div">{div_plot}</div>
            </div>
            <script>
                var data = {data_json};
                var metricNames = {metric_names_json};
                var datasetNames = {dataset_names_json};

                function getWeights() {{
                    return {{
                        {weight_readers_js}
                    }};
                }}

                function updateWeights() {{
                    // Update displayed values
                    {value_updaters_js}

                    var dsName = document.getElementById('dataset-select').value;
                    var weights = getWeights();
                    var configs = data[dsName];

                    // Calculate weighted scores
                    var scored = configs.map(function(c) {{
                        var totalWeight = 0;
                        var weightedSum = 0;
                        for (var m in c.metrics) {{
                            var w = weights[m] || 0;
                            totalWeight += w;
                            weightedSum += c.metrics[m] * w;
                        }}
                        var score = totalWeight > 0 ? weightedSum / totalWeight : 0;
                        return {{label: c.label, score: score}};
                    }});

                    // Sort by score (lower is better)
                    scored.sort(function(a, b) {{ return a.score - b.score; }});

                    // Limit to top 20
                    scored = scored.slice(0, 20);

                    var labels = scored.map(function(s) {{ return s.label; }});
                    var values = scored.map(function(s) {{ return s.score; }});
                    var colors = scored.map(function(s, i) {{
                        return i === 0 ? 'gold' : 'steelblue';
                    }});

                    var title = scored.length > 0
                        ? 'Best: ' + scored[0].label + ' (score: ' + scored[0].score.toExponential(4) + ')'
                        : 'Configuration Comparison';

                    Plotly.restyle('config_plot', {{
                        x: [labels],
                        y: [values],
                        'marker.color': [colors]
                    }});

                    Plotly.relayout('config_plot', {{title: title}});
                }}

                function resetWeights() {{
                    metricNames.forEach(function(m) {{
                        var id = m.toLowerCase().replace(' ', '_') + '-weight';
                        var el = document.getElementById(id);
                        el.value = m === 'SRMSE' ? '1.0' : '0.0';
                    }});
                    updateWeights();
                }}

                // Initialize
                updateWeights();
            </script>
        </body>
        </html>
        """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
