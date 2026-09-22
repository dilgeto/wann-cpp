# l2f_ui — vendored from `arplaboratory/learning-to-fly`

Files `drone.js`, `default_model.js`, `simulator.js`, `coordinate_system.js`, `math.js`
are copied verbatim from:

  https://github.com/arplaboratory/learning-to-fly
  path: src/ui/static/
  commit: d07592d5c5dea3c90954d2be6f04cfa68581ebe8

Copyright (c) 2023 Jonas Eschmann — MIT License (see that repo's `LICENSE`).

`lib/three.module.js` and `lib/OrbitControls.js` are three.js r156, fetched the
same way `learning-to-fly/src/ui/get_dependencies.sh` does (MIT License,
Three.js Authors).

Used here by `replay_l2f_3d.html` to render a proper quadrotor mesh instead of
a placeholder, reading a local WANN L2F replay CSV (offline, no WebSocket) —
not the live training UI these files were originally built for.
