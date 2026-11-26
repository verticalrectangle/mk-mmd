# The paper-dawn look

A style guide: the rules of one house look, each with its reason, its numbers and the key that holds them. The look is cut
paper and die-cut stickers on a Rosé Pine Dawn page: a figure cut out of construction paper in a toy theatre, real hands
that reach in and change the picture, real footage recoloured into the palette behind her. Read it before planning a video
in the look, and put [the checklist](#a-briefs-checklist) in every worker's brief. [design.md](design.md) is the reference
for every key named here.

The rules were lifted from three finished videos. What all three set alike became the look; where they differed, the later
two were followed. What a video says is not in here and never should be: its cut, its hook, which effects it uses and in
what order, how its set is dressed. Fix those and every video is the same video.

## Taking the look

```toml
[project]
include = ["look:paper-dawn", "~/mk-assets/looks/my_model.paper-dawn.toml"]   # the look, then the character's colours in it
```

`look:paper-dawn` is `mkmmd/looks/paper-dawn.toml` ([Includes and looks](design.md#includes-and-looks)): the project's own
keys are laid over it. It sets:

| Key | Value | Rule |
|---|---|---|
| `[look] palette` | `"rose-pine-dawn"` | 1 |
| `[vector] colors` | `{ background = "surface", line = "text" }`: the page and the ink | 1 |
| `[vector] light` | `[0.35, -0.55, 0.76]`: above, in front, from the figure's left | 2 |
| `[vector.tones]` | `skin = { lit = "#fbe9e0", shade = "#f1cfc6" }`, `whites = "surface"`, `ink = "text"`, `mouth = { lit = "love", shade = "love:text:0.3" }`, `tongue = "rose"`, `dark = "text"`, `thread = "text"`, `set = { texture = true }` | 2, 6 |
| `[post]` | `saturation = 1.0`, `contrast = 1.0`, `vignette = 0.0`, `floor = false`, `grain = { amount = 0.0 }` | 8 |

The rest of the look is the library's own defaults: a project gets them by writing nothing.

| What | Default | Where it lives |
|---|---|---|
| a sticker's rim | white, 16 px at 1080 on the short side | `mkmmd/core/sheet.py` `EDGE`: sheets and cut-out hands (a letter sticker's `rim` is the same 16) |
| a sticker's shadow | `text`, offset `[0.004, 0.007]` frame heights, blur 6 px, alpha 0.3 | `sheet.py` `SHADOW`: sheets and cut-out hands (craft type's own is the same) |
| thrown stickers | rim white 12 px, shadow alpha 0.25 | `sheet.py` `FLING` |
| the paper cut | grain 0.6, the scissors wandering 1.6 px, pieces lifted 3 px, cut again 8 times a second | `mkmmd/core/cutout.py` |
| poses | 12 a second | `[[stepped]] fps`, a cut-out hand's `fps` |
| footage ramps | cool: `text`, `pine`, `foam`, pale foam, `surface`; warm: `text`, `love`, `rose`, pale gold, `surface`; `warmth` 2.2 | `mkmmd/core/table.py` `RAMPS`, `WARMTH` |
| a hand clip's skin | the grade stored with the clip | the asset registry, the clip's `extra.grade` |

A project still writes what is its own: `style = "vector"` and `cutout = true` on each shot, its `[[stepped]]` windows, its
set's `step = 12`, the material rules of its set and props, a smaller sticker's rim (rule 3), each footage clip's `lean`.

A character's file holds what is the character's: its own tones (hair, eyes, costume, in palette slots) and the
`[[vector.materials]]` rules that give every material of the model its tone, the look's shared tones included (`skin`,
`whites`, `ink`, `mouth`, `tongue`, `dark`). It lives with the model's other data, outside this repository, and is
included after the look. A project's own rules are read before an included file's, so a video can give one material
another tone without touching the file.

## The rules

### 1. Colour: Rosé Pine Dawn by slot, and nothing black

- Every colour is a palette slot or a mix of two (`"love:text:0.3"`), by role: paper `base` and `surface`, shade and
  creases `overlay`, ink and shadows `text`, pops `love`, `gold`, `rose`, `pine`, `foam`, `iris`. The page
  (`[vector] colors.background`) is `surface`, the lightest paper; ink is `text`, the palette's darkest colour.
- Nothing is black. The darkest thing in the frame is `text`, and a shadow is `text` at 0.3. Night stays Dawn: plum paper
  (the dark end of every ramp is `text`), never a dark palette.
- Outside the palette there are only a character's own colours, the two skin papers and the skin of the real hands.

Why: the palette is what makes paper, footage and photographed hands one picture. Dawn has no black in it, so a black
shadow or a black night reads as a hole in the page.

One exception stands as the three videos shipped it: the drop shadow that the cut-out look lays under the 3D figure is
the library's own (`cutout.shadow`: black at 0.35).

### 2. She is cut paper

- Every shot she is in has `style = "vector"` and `cutout = true`: no lit 3D, and no outlines (a cut-out draws none).
- A part is two papers, `{ lit, shade }`: lit and in shade under one light, `[vector] light`, fixed in the world, with the
  split at the look's default `shadow` (0.3). Small things are one flat colour (whites, ink, tongue, buttons).
- What the model draws on itself stays drawn. An iris is a drawn tone of four inks by the texture's brightness: ink, two of
  the eye's colours and the page, `at = [0.22, 0.5, 0.86]`, `grey = 0.2`.
- The paper is the cut-out look's default: mottled and fibred, its edge cut by wandering scissors, each piece lifted off the
  one under it and shadowing it, the cut made again 8 times a second.

Why: two papers and one shadow per part are how a collage shows a form, and one light fixed in the world keeps her the
same figure in every shot. The cut made again 8 times a second is a stop-motion boil: the paper stays alive while a pose
is held.

### 3. Whatever is laid on the page is a sticker

- It wears a white die-cut rim and casts a soft ink shadow down and to the right: her on a sheet or a table, a prop laid
  on its own, a hand, a doodle, a letter sticker.
- The rim goes by the sticker's size in the frame, in px at 1080 on the short side: 16 for a hand and for a figure as tall
  as the frame, 14 for the figure at half to three quarters of the frame's height, 12 at about a third and for a prop, 10
  at a quarter. Never thinner. Write `edge = { width = 14 }`; `thin = 6` leaves what is narrower than 6 px (her threads)
  bare of the rim.
- The shadow is the default: write nothing.

Why: the rim parts a sticker from whatever lies under it (footage, paper, another sticker), so she reads at any size, and
the shadow says she is laid on the page, not printed on it. A rim that stays 16 px round a small figure swallows it:
hence the sizes.

### 4. Hands are real

- A hand is `[[fx]] kind = "cutout"`: a photographed hand cut out by its matte like a photo in a zine. A rough matte is
  fine under the rim.
- It is a sticker (rule 3: the rim of 16 px and the shadow are its defaults) and it is paper (rule 5: 12 poses a second,
  its default `fps`).
- One person's skin. A clip's grade is stored once, with the clip: `mk assets set SLUG extra='{"grade": {...}}'`, matched by
  eye against the hands already stored, lifted toward `rose`. An entry that writes no `grade` takes it. An entry's own
  `grade` is for a reason of that one shot, written beside it (a palm's finger shadows lifted).
- Its arm leaves the picture. Size and place the hand so that the border of its clip's own frame, where the camera cut the
  arm, lies outside the picture, then `extend = false`. Most hands are larger than life: 0.45 to 0.9 of the frame's
  height, where a natural hand is 0.18 to 0.3.
- It does something. Every hand changes the picture (it pokes, peels, snips, catches, tapes), in and out fast, never just
  waving, and its contact is on the beat: `sync = [the video's second of the contact, the beat]`, the entry starting early.

Why: the hands are what makes the paper world somebody's: a person is making this picture as it plays. Skin graded
anew for each video drifts from clip to clip, and two hands in one frame then belong to two people.

### 5. Paper moves at 12 poses a second, everything else on every frame

- In a clip at 60 frames a second, held poses: her (`[[stepped]]` over every take, 12 by default), the set (`step = 12`,
  with `step_from` a stepped window's `from` so that set and cast change pose on the same frames), paper props
  (`slots.step = 12`) and hands. The paper's own cut boils at 8.
- On every frame: the camera, a sticker's travel across the page, footage, glitches, zaps, smears and trails.
- What lands on a moment is there on the step held on that moment's frame.

Why: a figure on 12 against a camera on 60 is an animated film's character on its background. Paper is moved by hand and
shot a pose at a time; what is digital has no hand in it. The two rates side by side are the look.

### 6. The world is a toy theatre

- The stage is `[[set]] kind = "paper_theatre"`: construction paper and card in the palette's colours. Its pieces and the
  paper props are drawn each in its own paper colour: the tone `set = { texture = true }`, given to them by a rule of the
  project (`match = "theatre_*"`), a special piece's rule before it.
- A puppet's threads are ink: the tone `thread = "text"`. A thread's radius goes by the shot, from 0.0005 m in a close
  plate under a long lens to 0.0015 m in a wide one.

Why: a stage of paper puts her, the hands and the stickers in one hand-made world with a front, wings and a floor, where
anything can be lowered in on a thread.

### 7. Footage is brought to the palette before it enters the picture

- A clip under her is graded by warmth through two ramps, the library's defaults: cool pixels through `text`, `pine`,
  `foam`, pale foam, `surface`; warm ones through `text`, `love`, `rose`, pale gold, `surface`. Each clip leans its own
  way: `grade = { lean = -0.25 }` (cool, down to -0.35) or `0.25` (warm, up to 0.45); a clip the default leaves too dark
  or too pale has its own `stretch`, the luma percentiles that go to the ramps' ends.
- One plain surface (a table top) goes through a luma map instead: `grade = { map, lumas, soften }`, its stops set by eye.
- She and the paper things are never regraded.
- The grade leaves footage as it was brought: saturation and contrast 1.0 (rule 8).

Why: the footage must be Dawn itself, not a photograph tinted toward it. Two ramps keep warm light and cool sky apart, so
a city at dusk keeps its light, and two cities can lean apart and bleed into each other in a smear.

### 8. The grade is flat

- `[post]` adds nothing: no vignette, no floor, no grain, no halation, split tone or streaks, saturation and contrast 1.0.
- A video of paper alone, with no footage in it, may lift saturation to 1.12 and contrast to 1.05, as the first two did.

Why: the page is one evenly lit sheet of paper. A vignette, grain or a lifted black would make it a photograph of one.

### 9. Type is hand-craft, or absent

- Each line is its own craft (`[[text]] craft`): embossed label tape (`emboss`, `love`, tilted 3 to 4 degrees), handwriting
  on masking tape (`masking`), a die-cut letter sticker (`sticker`), cut-paper labels (`cutpaper`), brush letters (`brush`
  with `{ color = "pine", rim = 14, dry = 0.12 }`).
- Whole lines, or two or three chunks: never word by word.
- A video may have no type at all.

Why: type is one more thing laid on the page by the same hands, so it is made of the same paper, tape and ink.

### 10. Two standing rules that are not the look

They are in every brief all the same.

- The first view is her: the first frame shows the character.
- Nothing arrives late. Whatever travels sets off early, so that it lands on its moment, never after it.

## Left to each video

The cut and the hook. Which effects (tears, zaps, smears, trails, kaleidoscopes, liquid light) and how large. The set's
dressing. A drawn thread's look (`kind = "thread"`: cream and waving over a dark table, ink and taut over footage), a
trail's inks, the puppet's control and its wood.

## A brief's checklist

- Rosé Pine Dawn, colours by slot. Nothing black: the darkest colour is `text`, a shadow is `text` at 0.3.
- She is cut paper in every shot: `style = "vector"`, `cutout = true`, two papers a part, one light.
- Whatever is laid on the page is a sticker: a white rim by its size (16, 14, 12, 10 px) and the default shadow.
- Hands are real, rimmed, on 12 poses, in their clip's stored skin; the arm leaves the picture; the contact is on the beat;
  a hand always does something.
- Paper moves at 12 poses a second (her, the set, props, hands); the camera and what is digital move on every frame.
- Footage is recoloured into the palette before it enters; the grade is flat and leaves it alone.
- Type is hand-craft or absent.
- The first view is her. Nothing arrives late.
