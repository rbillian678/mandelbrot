# pip install matplotlib numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from pathlib import Path
import mandelbrot 




def choose_iters(xmin, xmax):
    span = abs(xmax - xmin)
    # log-based scale; clamp within sensible bounds
    it = int(np.clip(50 + 60*np.log2(3.0/span), 100, 2500))
    return it

W, H = 800, 800
DEFAULT_BOUNDS = (-2.0, 1.0, -1.5, 1.5)
xmin, xmax, ymin, ymax = DEFAULT_BOUNDS
it = 1000

print(f"Computing initial image {W}x{H}, iters={it} ...")
Z = mandelbrot.mandelbrot_neon(xmin, xmax, ymin, ymax, W, H, it) # max_iter

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(
    Z,
    extent=[xmin, xmax, ymin, ymax],
    origin='lower',
    cmap='inferno',
    interpolation='nearest',
)
ax.set_title("Mandelbrot — use toolbar zoom/pan or mouse wheel (R=reset, S=save)")
ax.set_xlabel("Re(c)"); ax.set_ylabel("Im(c)")
ax.set_aspect('equal', adjustable='box')

last_xlim = ax.get_xlim()
last_ylim = ax.get_ylim()
last_update = 0.0
COOLDOWN = 0.15  # seconds; simple debounce so we don't recompute too often

def recompute_if_needed():
    global last_xlim, last_ylim, last_update

    # Debounce
    now = time.time()
    if now - last_update < COOLDOWN:
        return
    last_update = now

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # Only recompute if the limits changed a noticeable amount
    epsx = 1e-12 * max(1.0, abs(x1 - x0))
    epsy = 1e-12 * max(1.0, abs(y1 - y0))

    if (abs(x0 - last_xlim[0]) < epsx and abs(x1 - last_xlim[1]) < epsx and
        abs(y0 - last_ylim[0]) < epsy and abs(y1 - last_ylim[1]) < epsy):
        return

    # Update cache first so nested draw events don't spam
    last_xlim = (x0, x1)
    last_ylim = (y0, y1)

    print(f"Recomputing: [{x0:.9f}, {x1:.9f}] x [{y0:.9f}, {y1:.9f}], iters={it}")
    Z2 = mandelbrot.mandelbrot_neon(x0, x1, y0, y1, W, H, it) #iters

    # Update image + extent
    im.set_data(Z2)
    im.set_extent([x0, x1, y0, y1])
    fig.canvas.draw_idle()

def on_draw(event):
    # Fires after zoom/pan/scroll when figure redraws
    recompute_if_needed()

def on_button_release(event):
    # After using toolbar zoom box or pan, mouse release triggers
    if event.inaxes is ax:
        recompute_if_needed()

def on_key(event):
    if event.key in ('r', 'R'):
        # reset to default
        print("Resetting view")
        ax.set_xlim(DEFAULT_BOUNDS[0], DEFAULT_BOUNDS[1])
        ax.set_ylim(DEFAULT_BOUNDS[2], DEFAULT_BOUNDS[3])
        fig.canvas.draw_idle()  # will trigger on_draw -> recompute
    elif event.key in ('s', 'S'):
        out = Path("mandelbrot.png")
        plt.savefig(out, dpi=200)
        print(f"Saved {out.resolve()}")

# Connect events
fig.canvas.mpl_connect('draw_event', on_draw)
fig.canvas.mpl_connect('button_release_event', on_button_release)
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()