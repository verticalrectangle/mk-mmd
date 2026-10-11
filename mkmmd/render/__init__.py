"""mk's own renderer for the vector look (docs/design.md: Rendering): `[render] engine = "mk"`. `plan` picks the shots
it can draw from the scene's bake, `engine` draws their passes on the GPU in headless Chromium (the page in
mkmmd/site/web/js/render/), and `compose` finishes them with the same compose_vector Blender's frames are finished
with. mk render gives Blender every other shot and the layers next to the frames."""
