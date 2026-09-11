"""Interactive HTML: F(z) on the left, F(x, y) at the selected z on the right,
driven by a custom z slider."""
from __future__ import annotations

import json
import os

import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors
from plotly.subplots import make_subplots

FONT_FAMILY = "Georgia, 'Times New Roman', Times, serif"
FONT_SIZE = 15
AXIS_TITLE_FONT_SIZE = 18
SUBPLOT_TITLE_FONT_SIZE = 18
CMAP_N_STOPS = 32

OVERLAY_OPACITY = 0.7
BOND_COLOR = "rgb(90,90,90)"
BOND_WIDTH = 1.5
_MARKER_LINE = dict(color="black", width=0.5)
# Hand-picked styles; any other element falls back to Jmol colours sized by
# covalent radius.
ELEMENT_STYLE = {
    "C": dict(color="rgb(160,160,160)", size=9),
    "N": dict(color="rgb(78,105,249)", size=9),
    "H": dict(color="rgb(255,255,255)", size=4),
}

# Trace layout of the figure: 0 = F(z) curve, 1 = heatmap, 2 = bonds, 3.. = atoms
_HEATMAP_TRACE = 1


def element_marker(symbol):
    if symbol in ELEMENT_STYLE:
        style = dict(ELEMENT_STYLE[symbol])
    else:
        z = atomic_numbers[symbol]
        r, g, b = (int(255 * c) for c in jmol_colors[z])
        size = float(np.clip(9 * covalent_radii[z] / covalent_radii[6], 4, 14))
        style = dict(color=f"rgb({r},{g},{b})", size=size)
    style["line"] = _MARKER_LINE
    return style


def mpl_cmap_to_plotly(cmap_name, n_stops=CMAP_N_STOPS):
    """Sample a matplotlib colormap into a Plotly colorscale."""
    cmap = mpl.colormaps[cmap_name]
    stops = []
    for i in range(n_stops):
        frac = i / (n_stops - 1)
        r, g, b, _ = cmap(frac)
        stops.append([frac, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"])
    return stops


CONTROL_PANEL_JS = """
(function() {{
    var gd = document.getElementById('{plot_id}');
    var zValues = {z_values_json};
    var frameNames = {frame_names_json};
    var zIdx = {z_default_idx};
    var atomTraces = {atom_traces_json};

    var panelHtml =
        '<div>' +
        '<label for="zSlider" style="display:inline-block;width:220px;">z center: <b><span id="zVal"></span></b> &Aring;</label>' +
        '<input id="zSlider" type="range" min="0" max="' + (zValues.length - 1) + '" step="1" value="' + zIdx + '" style="width:420px;vertical-align:middle;">' +
        '<input id="zInput" type="number" min="' + zValues[0] + '" max="' + zValues[zValues.length - 1] + '" step="0.01" value="' + zValues[zIdx].toFixed(2) + '" style="width:80px;margin-left:14px;vertical-align:middle;"> &Aring;' +
        '</div>' +
        (atomTraces.length ?
        '<div style="margin-top:8px;">' +
        '<label for="atomsToggle" style="display:inline-block;width:220px;">' +
        '<input id="atomsToggle" type="checkbox" checked style="vertical-align:middle;margin-right:6px;">show structure overlay' +
        '</label>' +
        '</div>' : '');

    var panel = document.createElement('div');
    panel.style.cssText = 'max-width:900px;margin:8px auto 0 auto;font-size:14px;';
    panel.style.fontFamily = {font_family_json};
    panel.innerHTML = panelHtml;
    gd.parentNode.insertBefore(panel, gd.nextSibling);

    var zLabel = panel.querySelector('#zVal');
    var zSlider = panel.querySelector('#zSlider');
    var zInput = panel.querySelector('#zInput');
    var atomsToggle = panel.querySelector('#atomsToggle');

    function nearestIdx(val) {{
        var best = 0, bestDiff = Infinity;
        for (var i = 0; i < zValues.length; i++) {{
            var d = Math.abs(zValues[i] - val);
            if (d < bestDiff) {{ bestDiff = d; best = i; }}
        }}
        return best;
    }}

    function render() {{
        zLabel.textContent = zValues[zIdx].toFixed(2);
        zSlider.value = zIdx;
        zInput.value = zValues[zIdx].toFixed(2);
        Plotly.animate(gd, [frameNames[zIdx]], {{
            frame: {{duration: 0, redraw: true}},
            transition: {{duration: 0}},
            mode: 'immediate'
        }});
    }}

    zSlider.addEventListener('input', function(e) {{ zIdx = +e.target.value; render(); }});
    zInput.addEventListener('change', function(e) {{
        zIdx = nearestIdx(parseFloat(e.target.value));
        render();
    }});
    if (atomsToggle) {{
        atomsToggle.addEventListener('change', function(e) {{
            Plotly.restyle(gd, {{visible: e.target.checked}}, atomTraces);
        }});
    }}
    render();
}})();
"""


def build_figure(result, title="lateral STA force", cmap="afmhot"):
    s = result.settings
    hw = s.half_width
    z0 = float(result.z_values[result.z_default_idx])
    f_lim = float(np.nanmax(np.abs(result.f1d))) * 1.15

    fig = make_subplots(rows=1, cols=2, column_widths=[0.38, 0.62],
                        subplot_titles=("", title), horizontal_spacing=0.08)

    # shapes[0] is the z-slice band; build_frames moves it
    fig.add_shape(type="rect", xref="x", yref="y", x0=z0 - hw, x1=z0 + hw,
                  y0=-f_lim, y1=f_lim, fillcolor="rgba(120,120,120,0.25)",
                  line=dict(width=0), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.z_mid, y=result.f1d, mode="lines",
                             line=dict(color="black", width=1.5), name="F(z)",
                             showlegend=False), row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color="gray", width=0.8), row=1, col=1)
    fig.add_vline(x=result.z_default, line=dict(color="crimson", dash="dash", width=0.8),
                  row=1, col=1)
    fig.update_xaxes(title_text="Height above local surface (Å)", range=[0, s.z_max],
                     row=1, col=1)
    fig.update_yaxes(title_text="<i>F(z)</i>  (eV/Å)", range=[-f_lim, f_lim], row=1, col=1)

    fig.add_trace(go.Heatmap(z=result.slice_map(z0), x=result.x, y=result.y,
                             colorscale=mpl_cmap_to_plotly(cmap),
                             zmin=result.vmin, zmax=result.vmax,
                             colorbar=dict(title="<i>F</i> (eV/Å)", x=1.02)),
                  row=1, col=2)

    ov = result.overlay
    if ov is not None:
        bx, by = [], []
        for i, j in ov.bonds:
            bx += [ov.xy[i, 0], ov.xy[j, 0], None]
            by += [ov.xy[i, 1], ov.xy[j, 1], None]
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines",
                                 line=dict(color=BOND_COLOR, width=BOND_WIDTH),
                                 opacity=OVERLAY_OPACITY, hoverinfo="skip",
                                 showlegend=False, name="bonds"), row=1, col=2)
        # by atomic number, but H last so the small H markers stay on top
        elements = sorted(set(ov.symbols), key=lambda e: (e == "H", atomic_numbers[e]))
        for elem in elements:
            sel = ov.symbols == elem
            fig.add_trace(go.Scatter(x=ov.xy[sel, 0], y=ov.xy[sel, 1], mode="markers",
                                     marker=element_marker(elem), opacity=OVERLAY_OPACITY,
                                     name=elem), row=1, col=2)

    a_vec, b_vec = result.cell2d
    c = [np.zeros(2), a_vec, a_vec + b_vec, b_vec]
    path = "M " + " L ".join(f"{p[0]},{p[1]}" for p in c) + " Z"
    fig.add_shape(type="path", xref="x2", yref="y2", path=path,
                  line=dict(color="black", width=1, dash="dash"))

    fig.update_xaxes(title_text="", showticklabels=False, ticks="", row=1, col=2,
                     scaleanchor="y2", scaleratio=1, range=[result.x[0], result.x[-1]])
    fig.update_yaxes(title_text="", showticklabels=False, ticks="", row=1, col=2,
                     range=[result.y[0], result.y[-1]])

    fig.update_layout(height=560, width=1150, title=None, template="plotly_white",
                      legend=dict(x=1.16, y=1.0),
                      font=dict(family=FONT_FAMILY, size=FONT_SIZE))
    fig.update_annotations(font=dict(family=FONT_FAMILY, size=SUBPLOT_TITLE_FONT_SIZE))
    axis_fonts = dict(title_font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE),
                      tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE))
    fig.update_xaxes(**axis_fonts)
    fig.update_yaxes(**axis_fonts)
    return fig


def build_frames(fig, result):
    hw = result.settings.half_width
    base_shapes = [sh.to_plotly_json() for sh in fig.layout.shapes]
    frames, names = [], []
    for zi, z0 in enumerate(result.z_values):
        shapes = [dict(sh) for sh in base_shapes]
        shapes[0]["x0"] = float(z0) - hw
        shapes[0]["x1"] = float(z0) + hw
        name = f"z{zi}"
        # zmin/zmax are not set per frame: the colour scale stays fixed
        frames.append(go.Frame(name=name, data=[go.Heatmap(z=result.slice_map(float(z0)))],
                               traces=[_HEATMAP_TRACE], layout=go.Layout(shapes=shapes)))
        names.append(name)
    return frames, names


def write_html(result, path, title="lateral STA force", cmap="afmhot", plotlyjs="inline"):
    """Write the interactive force map to `path`. plotlyjs="cdn" makes the
    file ~4.5 MB smaller but needs internet access to display."""
    fig = build_figure(result, title=title, cmap=cmap)
    frames, names = build_frames(fig, result)
    fig.frames = frames
    n_traces = len(fig.data)
    post_script = CONTROL_PANEL_JS.format(
        plot_id="{plot_id}",
        z_values_json=json.dumps([round(float(v), 3) for v in result.z_values]),
        frame_names_json=json.dumps(names),
        z_default_idx=result.z_default_idx,
        atom_traces_json=json.dumps(list(range(2, n_traces)) if result.overlay else []),
        font_family_json=json.dumps(FONT_FAMILY),
    )
    fig.write_html(path, include_plotlyjs={"inline": True, "cdn": "cdn"}[plotlyjs],
                   full_html=True, auto_play=False, post_script=post_script)
    return os.path.getsize(path)
