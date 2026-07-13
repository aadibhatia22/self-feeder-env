# Food models

These are lightweight, primitive-only MuJoCo food assets. Each category has
its own directory and can be loaded as a standalone model or attached to a
scene with `<model>` and `<attach>`.

| Directory | Asset | Contents |
|---|---|---|
| `fruit_slices/` | `fruit_slices.xml` | apple and orange slices |
| `tofu/` | `tofu.xml` | two tofu blocks |
| `vegetables/` | `vegetables.xml` | carrot, broccoli, and cucumber slice |
| `chicken/` | `chicken.xml` | chicken breast and wing |

All dimensions are in metres. Primitive geometry keeps the models easy to
scale, simulate, and replace with detailed meshes later.
