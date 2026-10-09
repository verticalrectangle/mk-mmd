--!strict
-- Block mk.pmx: a .pmx with its numbers, posed (T-pose first, then any pose its bones allow, with or without one of its
-- morphs) and shown in Tern's 3D block beside it, plus a lab sheet of the pose (toon views at one scale, numbers under
-- them). `mk model glb` reads, poses and writes the model (no Blender); mk://preview opens the .glb in the 3D block.

local ui = tern.ui
local L = require("./lib")

local TIMEOUT_MS = 300000
local POSE_NAMES = {
	tpose = "T-pose",
	rest = "Rest",
	arms_down = "Arms down",
	sit = "Sit",
	relaxed = "Relaxed hands",
	curled = "Curled hands",
	fist = "Fists",
	spread = "Spread hands",
}
local POSE_KEYS = { t = "tpose", r = "rest", a = "arms_down", s = "sit", e = "relaxed", c = "curled", f = "fist", p = "spread" }
local PANELS = { "eye", "eyebrow", "mouth", "other" }
local POSE_ORDER = { "tpose", "rest", "arms_down", "sit", "relaxed", "curled", "fist", "spread" }

type Morph = { name: string, panel: string }
type Info = {
	name: string,
	name_en: string,
	height_m: number,
	vertices: number,
	triangles: number,
	materials: number,
	drawn_materials: number,
	bones: number,
	morphs: { Morph },
	poses: { string },
}
type State = {
	cx: BlockCx,
	path: string?,
	name: string,
	info: Info?,
	err: string?,
	pose: string,
	morph: number?,
	busy: string?,
	glbs: { [string]: string },
	took: number?,
	sheet: string?,
	sheet_pose: string?,
	sheet_busy: boolean,
}

local live: { [number]: State } = {}

local function morph_name(state: State, i: number?): string?
	local info = state.info
	if i == nil or info == nil then
		return nil
	end
	local m = info.morphs[i :: number]
	return if m ~= nil then m.name else nil
end

local function shown_key(state: State): string
	return state.pose .. "|" .. (morph_name(state, state.morph) or "")
end

-- Pose the model (once per pose and morph in this run) and show it in the 3D block.
local function show(state: State)
	local cx = state.cx
	local path = state.path
	if path == nil then
		return
	end
	local k = shown_key(state)
	local done = state.glbs[k]
	if done ~= nil and tern.fs.exists(done) then
		cx:open(L.link("preview", { path = done }))
		return
	end
	local dir = tern.plugin.data .. "/pmx"
	pcall(tern.fs.mkdir, dir)
	local out = dir .. "/" .. L.hash({ path, k }) .. ".glb"
	local args = { "model", "glb", path, "--pose", state.pose, "--out", out }
	local morph = morph_name(state, state.morph)
	if morph ~= nil then
		table.insert(args, "--morph")
		table.insert(args, morph)
	end
	state.busy = (POSE_NAMES[state.pose] or state.pose) .. (if morph ~= nil then " + " .. morph else "")
	local t0 = tern.now()
	local ok, problem = pcall(function()
		tern.process.run(L.argv(args), { timeout_ms = TIMEOUT_MS }, function(r)
			if live[cx.pane] ~= state then
				return
			end
			state.busy = nil
			local rep, err = L.report(r)
			if rep == nil then
				state.err = err
			else
				state.err = nil
				if state.info == nil then
					state.info = rep :: Info
				end
				state.glbs[k] = out
				state.took = (tern.now() - t0) / 1000
				if shown_key(state) == k then
					cx:open(L.link("preview", { path = out }))
				end
			end
			cx:render()
		end)
	end)
	if not ok then
		state.busy = nil
		state.err = "cannot run mk: " .. tostring(problem)
	end
end

-- A lab sheet of the pose: front, side, back and three-quarter toon views with the numbers under them.
local function sheet(state: State)
	local cx = state.cx
	local path = state.path
	if path == nil or state.sheet_busy then
		return
	end
	local pose = state.pose
	local out = tern.plugin.data .. "/pmx/" .. L.hash({ path, "lab", pose }) .. ".png"
	pcall(tern.fs.mkdir, tern.plugin.data .. "/pmx")
	state.sheet_busy = true
	local ok = pcall(function()
		tern.process.run(
			L.argv({ "model", "lab", path, "--poses", pose, "--views", "front,outer,back,3q", "--size", "300", "--out", out }),
			{ timeout_ms = TIMEOUT_MS },
			function(r)
				if live[cx.pane] ~= state then
					return
				end
				state.sheet_busy = false
				local rep = L.report(r)
				if rep ~= nil and tern.fs.exists(out) then
					state.sheet = cx:blob(tern.fs.read(out), "image/png")
					state.sheet_pose = pose
				end
				if state.sheet_pose ~= state.pose then
					sheet(state)
				end
				cx:render()
			end
		)
	end)
	if not ok then
		state.sheet_busy = false
	end
end

local function set_pose(state: State, pose: string)
	local info = state.info
	if info == nil or pose == state.pose or not L.has(info.poses, pose) then
		return
	end
	state.pose = pose
	show(state)
	sheet(state)
end

local function set_morph(state: State, i: number?)
	state.morph = if i == state.morph then nil else i
	show(state)
end

local block: BlockDef<State> = {
	init = function(cx, args, saved)
		local s: any = if type(saved) == "table" then saved else {}
		local path: string? = if type(s.path) == "string" then s.path else args[1]
		local state: State = {
			cx = cx,
			path = path,
			name = if path ~= nil then string.match(path :: string, "([^/]+)$") or (path :: string) else "PMX",
			info = nil,
			err = nil,
			pose = if type(s.pose) == "string" then s.pose else "tpose",
			morph = nil,
			busy = nil,
			glbs = {},
			took = nil,
			sheet = nil,
			sheet_pose = nil,
			sheet_busy = false,
		}
		live[cx.pane] = state
		if path ~= nil and not tern.fs.exists(path :: string) then
			state.err = "no file " .. (path :: string)
		elseif path ~= nil then
			show(state)
			sheet(state)
		end
		return state
	end,

	title = function(state)
		local info = state.info
		return (if info ~= nil and info.name ~= "" then info.name else state.name) .. " · " .. (POSE_NAMES[state.pose] or state.pose)
	end,

	view = function(state, _cx)
		if state.path == nil then
			return { main = ui.col({ ui.text({ ui.span("Open a .pmx file to see it here", "muted") }) }) }
		end
		local kids: { Node } = {}
		local info = state.info
		local title: { Span | string } = { ui.span(if info ~= nil and info.name ~= "" then info.name else state.name, "strong") }
		if info ~= nil and info.name_en ~= "" and info.name_en ~= info.name then
			table.insert(title, ui.span("  " .. info.name_en, "muted"))
		end
		table.insert(kids, L.props(ui.text(title), { key = "title" }))
		table.insert(kids, L.props(ui.text({ ui.path(state.path :: string) }), { key = "path" }))
		if info ~= nil then
			table.insert(kids, L.props(ui.kv({
				{ "Height", string.format("%.3f m", info.height_m) },
				{ "Vertices", string.format("%d (%d triangles)", info.vertices, info.triangles) },
				{ "Materials", string.format("%d (%d drawn)", info.materials, info.drawn_materials) },
				{ "Bones", tostring(info.bones) },
				{ "Morphs", tostring(#info.morphs) },
			}), { key = "numbers" }))
			local poses: { Node } = {}
			for _, p in POSE_ORDER do                    -- the poses this model's bones allow, whole-body ones first
				if L.has(info.poses, p) then
					table.insert(poses, L.button(POSE_NAMES[p] or p, "pose=" .. p, if p == state.pose then "accent" else "neutral"))
				end
			end
			table.insert(kids, L.props(ui.section({ ui.span("Pose", "strong") }, { L.props(ui.row(poses), { role = "mk.chips" }) }), { key = "poses" }))
			for _, panel in PANELS do
				local chips: { Node } = {}
				for i, m in info.morphs do
					if m.panel == panel then
						table.insert(chips, L.button(m.name, "morph=" .. i, if i == state.morph then "accent" else "neutral", "m" .. i))
					end
				end
				if #chips > 0 then
					table.insert(kids, L.props(ui.section({ ui.span("Morphs: " .. panel, "strong") },
						{ L.props(ui.row(chips), { role = "mk.chips" }) }), { key = "morphs-" .. panel }))
				end
			end
		end
		local status: Node
		if state.busy ~= nil then
			status = ui.node("spinner", { style = "dots", label = { ui.span("Posing " .. (state.busy :: string) .. "…", "muted") } })
		elseif state.err ~= nil then
			status = ui.code(state.err :: string, "text")
		elseif state.took ~= nil then
			status = ui.text({ ui.span(string.format("In the 3D block beside this one (posed in %.1f s)", state.took :: number), "muted") })
		else
			status = ui.text({ ui.span("", "muted") })
		end
		table.insert(kids, L.props(status, { key = "status" }))
		if state.sheet ~= nil then
			table.insert(kids, L.props(ui.node("image", { blob = state.sheet, alt = "Lab sheet: " .. state.name, max = { h = "34lines" } }), { key = "sheet" }))
		elseif state.sheet_busy then
			table.insert(kids, L.props(ui.node("spinner", { style = "dots", label = { ui.span("Drawing the lab sheet…", "muted") } }), { key = "sheet" }))
		end
		return {
			main = ui.col(kids),
			dock = ui.col({
				L.props(ui.row({ L.button("Show in 3D", "open", "accent"), L.button("No morph", "morph=0") }), { key = "bar" }),
				L.props(ui.text({ ui.span("t T-pose · r rest · a arms down · s sit · e/c/f/p hands · 0 no morph · o show in 3D", "dim") }), { key = "keys" }),
			}),
		}
	end,

	key = function(state, key, _cx)
		local name = key.name
		if POSE_KEYS[name] ~= nil then
			set_pose(state, POSE_KEYS[name])
		elseif name == "0" then
			if state.morph == nil then
				return false
			end
			set_morph(state, nil)
		elseif name == "o" then
			show(state)
		else
			return false
		end
		return true
	end,

	event = function(state, ev, _cx)
		if ev.ev ~= "action" then
			return
		end
		local value = if type(ev.value) == "string" then ev.value else ""
		if ev.act == "pose" then
			set_pose(state, value)
		elseif ev.act == "morph" then
			local i = tonumber(value)
			set_morph(state, if i ~= nil and i > 0 then i else nil)
		elseif ev.act == "open" then
			show(state)
		end
	end,

	save = function(state)
		return { path = state.path, pose = state.pose }
	end,
}

return block
