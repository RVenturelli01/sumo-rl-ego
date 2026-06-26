from abc import ABC, abstractmethod


class BaseStepDisplay(ABC):
    """
    Extension point for visualizing obs/action/reward during simulation.
    Subclass and pass an instance to play_policy(display=...).

    Hooks called by play_policy:
      on_episode_start(episode)           — before the episode loop
      on_step(obs, action, step)          — before env.step
      on_reward(reward, info, step)       — after env.step (default: no-op)
      on_episode_end(info)                — after the episode loop
    """

    def on_episode_start(self, episode: int):
        pass

    def on_episode_end(self, info: dict):
        pass

    @abstractmethod
    def on_step(self, obs, action, step: int):
        pass

    def on_reward(self, reward: float, info: dict, step: int):
        pass


class TerminalDisplay(BaseStepDisplay):
    """
    Prints action, obs and reward to the terminal at each step.

    Args:
        env: the SumoEnv instance.
        pause: if True, waits for Enter between steps.
        every_n_steps: display every N steps.
    """

    def __init__(self, env, pause: bool = False, every_n_steps: int = 1):
        self.env = env
        self.pause = pause
        self.every_n_steps = every_n_steps
        self._step = 0

    def on_episode_start(self, episode: int):
        print(f"\n{'=' * 48}")
        print(f"Episode {episode}")
        print(f"{'=' * 48}")

    def on_step(self, obs, action, step: int):
        self._step = step
        if step % self.every_n_steps != 0:
            return

        print(f"{'=' * 20} ACTION {'=' * 20}")
        print(self.env.ego_controller.format_action(action))

        print(f"{'=' * 20} OBS {'=' * 23}")
        print(self.env.obs_builder.format_obs(obs))

    def on_reward(self, reward: float, info: dict, step: int):
        if step % self.every_n_steps != 0:
            return

        print(f"{'=' * 20} REWARD {'=' * 20}")
        print(f"{reward:.4f}")
        print("=" * 48)

        if self.pause:
            input("Press Enter to step...\n")

    def on_episode_end(self, info: dict):
        if self.pause:
            input("Press Enter to start next episode...\n")


class WindowDisplay(BaseStepDisplay):
    """
    PySide6 + PyQtGraph real-time dashboard.

    Layout (all panels resizable via drag handles, zero overlap):
      Left column           │  Right column
      ──────────────────────┼─────────────────────────
      ACTION  (text)        │  Action chart
      OBSERVATION (text)    │  Step Reward chart
      REWARD FILTER (cbs)   │  Cumulative Reward chart

    Chart controls:
      Left-drag  → box zoom      Right-drag  → pan
      Scroll     → zoom X axis   Right-click → "View All" resets

    Requires: pip install PySide6 pyqtgraph
    If Qt fails to start: sudo apt install libxcb-cursor0
    """

    _COLORS = [
        (220,  53,  69),
        ( 25, 135,  84),
        ( 13, 110, 253),
        (253, 126,  20),
        (111,  66, 193),
        (  0, 180, 210),
        (231, 111,  81),
        ( 42, 157, 143),
        (180, 155,  40),
        (100, 160, 220),
    ]

    def __init__(self, env, pause: bool = False, every_n_steps: int = 1):
        self.env           = env
        self.pause         = pause
        self.every_n_steps = every_n_steps

        self._episode  = 0
        self._proceed  = False

        self._discrete      = self._detect_discrete()
        self._action_labels = self._get_action_labels() if self._discrete else []

        # histories reset each episode
        self._action_history: list = []
        self._rwd_histories:  dict = {}   # key -> list[float]

        # plot series — created once, reused across episodes
        self._act_curves:  list = []
        self._rwd_curves:  dict = {}
        self._cum_curves:  dict = {}
        self._checkboxes:  dict = {}
        self._series_ready = False

        self._fix_qt_env()
        self._build_window()

    # ──────────────────────────────────────────────────────────────────
    # Qt environment setup  (must run before QApplication is created)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_qt_env():
        import os
        import ctypes
        import ctypes.util

        # Override plugin path so cv2's Qt5 plugins don't shadow PySide6's Qt6 ones
        try:
            import PySide6 as _p6
            _plugins = os.path.join(os.path.dirname(_p6.__file__), 'Qt', 'plugins', 'platforms')
            if os.path.isdir(_plugins):
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = _plugins
        except ImportError:
            pass

        os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

        # Preload libxcb-cursor (required by Qt6's xcb platform plugin).
        # Try system install first, then the deb-extracted copy in /tmp.
        _candidates = [
            '/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0',
            '/usr/lib/aarch64-linux-gnu/libxcb-cursor.so.0',
            '/tmp/xcb-cursor/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0',
        ]
        for _path in _candidates:
            if os.path.exists(_path):
                try:
                    ctypes.CDLL(_path, ctypes.RTLD_GLOBAL)
                    return
                except OSError:
                    pass

        _lib = ctypes.util.find_library('xcb-cursor')
        if _lib:
            try:
                ctypes.CDLL(_lib, ctypes.RTLD_GLOBAL)
            except OSError:
                pass

    # ──────────────────────────────────────────────────────────────────
    # Action space helpers
    # ──────────────────────────────────────────────────────────────────

    def _detect_discrete(self) -> bool:
        from gymnasium.spaces import Discrete
        return isinstance(self.env.action_space, Discrete)

    def _get_action_labels(self) -> list:
        labels = []
        for i in range(self.env.action_space.n):
            try:
                labels.append(self.env.ego_controller.format_action(i))
            except Exception:
                labels.append(str(i))
        return labels

    @staticmethod
    def _fmt_key(key: str) -> str:
        import re
        s = key.replace('rewards/', '').replace('ep_', '').replace('_', ' ')
        s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)
        return s.strip().title()

    # ──────────────────────────────────────────────────────────────────
    # Window construction
    # ──────────────────────────────────────────────────────────────────

    def _build_window(self):
        try:
            import pyqtgraph as pg
            from PySide6.QtWidgets import (
                QApplication, QMainWindow, QWidget, QVBoxLayout,
                QSplitter, QGroupBox, QPlainTextEdit, QLabel,
                QScrollArea, QFrame,
            )
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QFont, QKeySequence, QShortcut
        except ImportError as exc:
            raise ImportError(
                "WindowDisplay requires PySide6 and pyqtgraph.\n"
                "Install:  pip install PySide6 pyqtgraph\n"
                f"Missing:  {exc}"
            ) from exc

        import sys

        pg.setConfigOptions(background='w', foreground='k', antialias=True)

        self._app = QApplication.instance() or QApplication(sys.argv)
        self._app.setStyle('Fusion')

        # main window
        self._win = QMainWindow()
        self._win.setWindowTitle('Agent Monitor')
        self._win.resize(1400, 900)

        root = QWidget()
        self._win.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(4)

        # header
        self._lbl_header = QLabel('Episode 0 · Step 0')
        hf = QFont()
        hf.setPointSize(13)
        hf.setBold(True)
        self._lbl_header.setFont(hf)
        self._lbl_header.setStyleSheet('color: #0d6efd; padding: 2px 6px;')
        vbox.addWidget(self._lbl_header)

        self._lbl_hint = None
        if self.pause:
            self._lbl_hint = QLabel('Space / Enter to step')
            hf2 = QFont()
            hf2.setPointSize(10)
            hf2.setItalic(True)
            self._lbl_hint.setFont(hf2)
            self._lbl_hint.setStyleSheet('color: #6c757d; padding: 0 6px 2px;')
            vbox.addWidget(self._lbl_hint)
            for key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
                sc = QShortcut(QKeySequence(key), self._win)
                sc.activated.connect(lambda: setattr(self, '_proceed', True))

        # outer horizontal splitter
        h_split = QSplitter(Qt.Horizontal)
        h_split.setHandleWidth(7)
        vbox.addWidget(h_split, stretch=1)

        # ── left column ───────────────────────────────────────────────
        left = QSplitter(Qt.Vertical)
        left.setHandleWidth(7)
        h_split.addWidget(left)

        mono = QFont('Courier', 11)
        _gs = (
            'QGroupBox {{ font-size:13px; font-weight:bold; color:{c}; '
            'border:1px solid #dee2e6; border-radius:4px; '
            'margin-top:10px; padding-top:4px; }}'
            'QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; }}'
        )

        def _text_panel(title, color):
            box = QGroupBox(title)
            box.setStyleSheet(_gs.format(c=color))
            lay = QVBoxLayout(box)
            lay.setContentsMargins(4, 14, 4, 4)
            te = QPlainTextEdit()
            te.setReadOnly(True)
            te.setFont(mono)
            lay.addWidget(te)
            return box, te

        ag, self._te_action = _text_panel('ACTION',      '#16a34a')
        og, self._te_obs    = _text_panel('OBSERVATION', '#b45309')
        left.addWidget(ag)
        left.addWidget(og)

        # reward filter panel
        fg = QGroupBox('REWARD FILTER')
        fg.setStyleSheet(_gs.format(c='#0d6efd'))
        fl = QVBoxLayout(fg)
        fl.setContentsMargins(4, 14, 4, 4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._filter_inner = QWidget()
        self._filter_vbox  = QVBoxLayout(self._filter_inner)
        self._filter_vbox.setAlignment(Qt.AlignTop)
        self._filter_vbox.setSpacing(3)
        self._lbl_filter_ph = QLabel('(rewards discovered at first step)')
        self._lbl_filter_ph.setStyleSheet('color:#6c757d; font-style:italic;')
        self._filter_vbox.addWidget(self._lbl_filter_ph)
        scroll.setWidget(self._filter_inner)
        fl.addWidget(scroll)
        left.addWidget(fg)

        left.setSizes([200, 380, 270])

        # ── right column (plots) ──────────────────────────────────────
        right = QSplitter(Qt.Vertical)
        right.setHandleWidth(7)
        h_split.addWidget(right)

        act_title = (
            f'Action  [{self.env.action_space.n} discrete]'
            if self._discrete else 'Action'
        )
        self._plot_act = self._make_plot(act_title,        'step', 'action')
        self._plot_rwd = self._make_plot('Step Reward',    'step', 'reward')
        self._plot_cum = self._make_plot('Cumul. Reward',  'step', 'cumulative')
        right.addWidget(self._plot_act)
        right.addWidget(self._plot_rwd)
        right.addWidget(self._plot_cum)
        right.setSizes([300, 300, 300])

        h_split.setSizes([420, 980])
        h_split.setStretchFactor(0, 1)
        h_split.setStretchFactor(1, 3)

        self._win.show()
        self._app.processEvents()

    @staticmethod
    def _make_plot(title: str, xlabel: str, ylabel: str):
        import pyqtgraph as pg
        pw = pg.PlotWidget(title=f'<b>{title}</b>')
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.setLabel('bottom', xlabel)
        pw.setLabel('left',   ylabel)
        pw.addLegend(offset=(10, 10))
        vb = pw.getViewBox()
        vb.setMouseMode(pg.ViewBox.RectMode)
        # PyQtGraph's built-in "View All" does a one-shot autoRange() that then
        # leaves autoRange *disabled*, so new data stops moving the axis.
        # Re-enable continuous tracking after the reset.
        try:
            vb.menu.viewAll.triggered.connect(lambda: vb.enableAutoRange())
        except AttributeError:
            pass
        return pw

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle hooks
    # ──────────────────────────────────────────────────────────────────

    def on_episode_start(self, episode: int):
        self._episode = episode
        self._action_history.clear()
        for k in self._rwd_histories:
            self._rwd_histories[k] = []

        self._lbl_header.setText(f'Episode {episode} · Step 0')

        for curve in self._act_curves:
            curve.setData([], [])
        for curve in self._rwd_curves.values():
            curve.setData([], [])
        for curve in self._cum_curves.values():
            curve.setData([], [])

        self._app.processEvents()

    def on_step(self, obs, action, step: int):
        import numpy as np
        self._action_history.append(np.atleast_1d(np.asarray(action, dtype=float)))

        if step % self.every_n_steps == 0:
            self._lbl_header.setText(f'Episode {self._episode} · Step {step}')
            self._te_action.setPlainText(self.env.ego_controller.format_action(action))
            self._te_obs.setPlainText(self.env.obs_builder.format_obs(obs))

        self._app.processEvents()

    def on_reward(self, reward: float, info: dict, step: int):
        step_metrics = info.get('metrics', {}).get('step', {})
        sub_keys = sorted(k for k in step_metrics if k.startswith('rewards/'))

        if not self._series_ready:
            all_keys = ['reward'] + sub_keys
            for k in all_keys:
                self._rwd_histories[k] = []
            self._init_reward_series(all_keys)

        self._rwd_histories['reward'].append(reward)
        for k in sub_keys:
            if k in self._rwd_histories:
                self._rwd_histories[k].append(step_metrics.get(k, 0.0))

        if step % self.every_n_steps == 0:
            self._update_charts()

        if self.pause:
            self._proceed = False
            while not self._proceed:
                self._app.processEvents()
        else:
            self._app.processEvents()

    def on_episode_end(self, info: dict):
        if self.pause and self._lbl_hint is not None:
            self._lbl_hint.setText('Episode ended — Space / Enter to continue')
            self._proceed = False
            while not self._proceed:
                self._app.processEvents()
            self._lbl_hint.setText('Space / Enter to step')
        else:
            self._app.processEvents()

    # ──────────────────────────────────────────────────────────────────
    # Reward series initialisation  (called once, on the first step)
    # ──────────────────────────────────────────────────────────────────

    def _init_reward_series(self, keys: list):
        import pyqtgraph as pg
        from PySide6.QtWidgets import QCheckBox
        from PySide6.QtGui import QFont

        if self._lbl_filter_ph is not None:
            self._lbl_filter_ph.setParent(None)
            self._lbl_filter_ph = None

        cb_font = QFont()
        cb_font.setPointSize(11)

        for i, k in enumerate(keys):
            r, g, b = self._COLORS[i % len(self._COLORS)]
            label   = 'Main Reward' if k == 'reward' else self._fmt_key(k)

            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setFont(cb_font)
            cb.setStyleSheet(f'color: rgb({r},{g},{b}); padding: 2px 6px;')
            cb.stateChanged.connect(self._update_charts)
            self._filter_vbox.addWidget(cb)
            self._checkboxes[k] = cb

            pen  = pg.mkPen(color=(r, g, b), width=2)
            fill = pg.mkBrush(r, g, b, 45)

            self._rwd_curves[k] = self._plot_rwd.plot([], [], pen=pen, name=label)
            self._cum_curves[k] = self._plot_cum.plot(
                [], [], pen=pen, fillLevel=0, brush=fill, name=label,
            )

        self._series_ready = True
        self._app.processEvents()

    # ──────────────────────────────────────────────────────────────────
    # Chart update
    # ──────────────────────────────────────────────────────────────────

    def _update_charts(self, *_):
        import numpy as np

        # action chart
        if self._action_history:
            act = np.array(self._action_history)   # shape (n, dims)
            xs  = list(range(len(act)))

            if not self._act_curves:
                import pyqtgraph as pg
                n_dims = act.shape[1]
                for i in range(n_dims):
                    r, g, b = self._COLORS[i % len(self._COLORS)]
                    name    = 'action' if n_dims == 1 else f'dim {i}'
                    self._act_curves.append(
                        self._plot_act.plot([], [], pen=pg.mkPen((r, g, b), width=2), name=name)
                    )
                if self._discrete:
                    n_ac = self.env.action_space.n
                    self._plot_act.setYRange(-0.5, n_ac - 0.5, padding=0)
                    self._plot_act.getAxis('left').setTicks(
                        [[(j, self._action_labels[j]) for j in range(n_ac)]]
                    )

            for i, curve in enumerate(self._act_curves):
                curve.setData(xs, act[:, i].tolist())

        # reward and cumulative reward charts
        for k in self._rwd_curves:
            visible = self._checkboxes[k].isChecked()
            vals    = self._rwd_histories.get(k, [])
            xs      = list(range(len(vals)))

            self._rwd_curves[k].setVisible(visible)
            self._cum_curves[k].setVisible(visible)

            if visible and vals:
                self._rwd_curves[k].setData(xs, vals)
                self._cum_curves[k].setData(xs, np.cumsum(vals).tolist())
