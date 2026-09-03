# Minecraft Resourcepack Tools
a collection of python tools for creating and editing minecraft block .json models , UVing , and creating animated textures  

## MODEL
- [mc_model_solver_ui](mc_model_solver_ui.py) Approximates 3D primitives (sphere/cone/pyramid/helix) with Minecraft 1.20 valid cuboid `elements` JSON.
- [mc_model_symmetry_fixer_ui](mc_model_symmetry_fixer_ui.py) Analyzes symmetry of a model and creates an IDEAL symmetric model JSON, fixing hand-adjustment drift.

- [mc_model_procedural_gen_ui](mc_model_procedural_gen_ui.py) Procedurally generates spiral/double-helix model JSON with radial arrangement and instances compositor (uses 
`mc_model_procedural_gen_common`, `_shape_helix`, `_shape_spiral`).

- [mc_diamond_pyramid_solver_ui](mc_diamond_pyramid_solver_ui.py) Builds diamond/bipyramid/pyramid models from thin-plane cuboids with alpha-cutout atlas support.

- [mc_build_design_planner_ui](mc_build_design_planner_ui.py) 2D pixel-shape planner (circles, double spirals) for laying out block builds; exports coords.

- [mc_model_greedy_optimizer_ui](mc_model_greedy_optimizer_ui.py) Reduces cuboid element count by merging pairs (including rotated ones sharing a frame) into single boxes, preserving exact volume (uses `mc_model_greedy_optimizer_core`).

## SHADER
- [mc_model_shade_false_enforcer_ui](mc_model_shade_false_enforcer_ui.py) Inserts and enforces `"shade": false` on every element of a Minecraft model JSON.

## UV
- [mc_model_uv_packer_ui](mc_model_uv_packer_ui.py) Box-unwraps cuboids into non-overlapping UV islands and packs them into an atlas with padding vertex snapping to specific texel coordinates for target texture size

## TEXTURE
- [mc_make_vertical_spritesheet_loop](mc_make_vertical_spritesheet_loop.py) Bakes a brightness/contrast loop into a PNG as a vertical animated spritesheet, with emissive mask and seamless scroll layer.

## RENDER
- [mc_resourcepack_blender_dashboard](mc_resourcepack_blender_dashboard.py) dashboard scans a resourcepack and runs headless Blender jobs to generate `.blend` caches and render PNGs.
- [mc_resourcepack_blender_generate_blends](mc_resourcepack_blender_generate_blends.py) Blender script: imports Minecraft model JSON / Optifine CEM `.jem` and saves a cached `.blend`.
- [mc_resourcepack_blender_render_blends](mc_resourcepack_blender_render_blends.py) Blender script: opens a cached `.blend` and renders one flat-textured PNG per camera.

## CEM
- [cem_obj_geom_diff](cem_obj_geom_diff.py) Compares two `.obj` files per-object (bbox/centroid/extents, optional exact vertex-set equivalence with translation/rotation detection).
- [cem_obj_uv_diff](cem_obj_uv_diff.py) Compares per-cuboid face UV rects and corner mapping between two `.obj` files to validate CEM exports against Blockbench.
