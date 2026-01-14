from typing import Dict

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

            # 1. Update Parity Plot
            # Flatten data for parity plot
            y_true_flat = y_true.values.flatten()
            y_pred_flat = y_pred.values.flatten()

            # Create sample indices for mapping back
            n_samples, n_points = y_true.shape
            sample_indices = np.repeat(np.arange(n_samples), n_points)

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
                        color=sample_indices,
                        colorscale='Turbo',
                        showscale=False
                    ),
                    hoverlabel=dict(bgcolor=hover_colors),
                    customdata=sample_indices,
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
                update_curve_plot(dataset_name, sample_idx)

            # Update Parity Plot Title with Global SRMSE
            srmse_pct = res['metrics'].get('SRMSE', 0) * 100
            parity_fig.update_layout(title=f"Parity Plot (Global SRMSE: {srmse_pct:.2f}%)")

            # Re-attach click callback to the new trace (only useful for curve datasets)
            if not is_scalar:
                parity_fig.data[0].on_click(on_parity_click)

        def update_curve_plot(dataset_name, sample_idx):
            res = self.results[dataset_name]
            y_pred: pd.DataFrame = res['y_pred']
            y_true: pd.DataFrame = res['y_true']
            y_pred_sample: pd.Series = y_pred.iloc[sample_idx]
            y_true_sample: pd.Series = y_true.iloc[sample_idx]

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
                curve_fig.update_layout(title=f"Sample {sample_idx} (SRMSE: {srmse_curve_pct:.2f}%)")

        # Callbacks
        def on_dataset_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                update_charts(change['new'])

        def on_parity_click(trace, points, selector):
            # Skip for scalar datasets - no curve to display
            if self._is_scalar(ds_dropdown.value):
                return
            if points.point_inds:
                # Get the sample index from customdata
                # points.point_inds[0] gives the index in the flattened array
                # We stored sample_indices in customdata
                flat_idx = points.point_inds[0]
                sample_idx = trace.customdata[flat_idx]
                update_curve_plot(ds_dropdown.value, sample_idx)

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

            # Scatter Trace
            # Create sample indices for coloring
            n_samples, n_points = res['y_true'].shape
            sample_indices = np.repeat(np.arange(n_samples), n_points)

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
                    color=sample_indices,
                    colorscale='Turbo',
                    showscale=False
                ),
                hoverlabel=dict(bgcolor=hover_colors),
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
        fig_curve.update_layout(
            title="Curve Comparison (Sample 0)",
            xaxis_title="Curve Point",
            yaxis_title="Value",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        for i, ds_name in enumerate(dataset_names):
            res = self.results[ds_name]
            # Only showing Sample 0 for HTML export simplicity
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
