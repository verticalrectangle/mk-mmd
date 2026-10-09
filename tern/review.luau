--!strict
-- Block mk.review: NAME.review.toml (mkmmd/review.py) as questions, each with its pictures, models to turn in Tern's
-- 3D block, a choice and notes. Choices and notes go to NAME.answers.json as they change; Send adds a message and has
-- the window half post it to the agent that opened the review (mk://send). Mark up opens a picture in a whiteboard to
-- draw on (mk://markup); Read marks collects what was drawn into NAME.marks.json (mk://marks), where `mk review
-- answers` reads lines drawn over lab sheets back in millimetres. `mk review show` reads and checks the file;
-- `mk model glb` poses .pmx and spec models.

local ui = tern.ui
local L = require("./lib")

local SHOW_TIMEOUT_MS = 120000
local GLB_TIMEOUT_MS = 600000

type Img = { id: string?, path: string, name: string, w: number, h: number, show: string, sheet: boolean }
type Model = { id: string?, label: string, open: string?, glb: { string }? }
type Option = { id: string, label: string, text: string, images: { Img }, models: { Model } }
type Question = {
	id: string,
	ask: string,
	text: string,
	recommended: string?,
	options: { Option },
	images: { Img },
	models: { Model },
}
type Review = {
	path: string,
	title: string,
	text: string,
	questions: { Question },
	answers_file: string,
	marks_file: string,
	reply_to: number?,
	answers: any,
	sent: string?,
}
type Answer = { choice: string?, notes: string }
type Built = { state: string, glb: string?, err: string? }
type State = {
	cx: BlockCx,
	path: string?,
	review: Review?,
	err: string?,
	loading: boolean,
	blobs: { [string]: string },
	pics: { [string]: Img },
	models: { [string]: Model },
	built: { [string]: Built },
	answers: { [string]: Answer },
	current: number,
	focus: string?,
	cursor: number,
	boards: { any },                 -- whiteboards opened on pictures: {pane, shape, image, w, h}
	marks: number?,
	sent: string?,
	status: string?,
}

-- The state of the block running in each pane: callbacks of an earlier run (before a reload) find another one there.
local live: { [number]: State } = {}

local function answer(state: State, qid: string): Answer
	local a = state.answers[qid]
	if a == nil then
		local fresh: Answer = { choice = nil, notes = "" }
		state.answers[qid] = fresh
		return fresh
	end
	return a
end

local function answered(state: State): (number, number)
	local review = state.review
	if review == nil then
		return 0, 0
	end
	local n, total = 0, 0
	for _, q in review.questions do
		if #q.options > 0 then
			total += 1
			local a = state.answers[q.id]
			if a ~= nil and a.choice ~= nil then
				n += 1
			end
		end
	end
	return n, total
end

-- The answers file as the person answers; `reply_to` and `review` are mk review open's and stay.
local function write_answers(state: State, extra: { [string]: any }?)
	local review = state.review
	if review == nil then
		return
	end
	local rec = L.read_json(review.answers_file)
	if type(rec) ~= "table" then
		rec = { review = review.path, reply_to = review.reply_to }
	end
	local out = {}
	for qid, a in state.answers do
		out[qid] = { choice = a.choice, notes = a.notes }
	end
	rec.answers = out
	for k, v in extra or {} do
		rec[k] = v
	end
	local ok, err = pcall(L.write_json, review.answers_file, rec)
	if not ok then
		state.status = "cannot write the answers: " .. tostring(err)
	end
end

local function label_of(q: Question, oid: string?): string?
	for _, o in q.options do
		if o.id == oid then
			return o.label
		end
	end
	return nil
end

-- What Send posts into the agent's chat.
local function message(state: State, review: Review): string
	local lines = { "Review answered: " .. review.title .. " (" .. review.path .. ")" }
	for i, q in review.questions do
		local a = state.answers[q.id]
		local pick = "no choice"
		if a ~= nil and a.choice ~= nil then
			pick = a.choice .. ": " .. (label_of(q, a.choice) or "?")
		end
		table.insert(lines, string.format("%d. %s → %s", i, q.ask, pick))
		if a ~= nil and a.notes ~= "" then
			table.insert(lines, "   notes: " .. (string.gsub(a.notes, "\n", "\n   ")))
		end
	end
	if state.marks ~= nil and (state.marks :: number) > 0 then
		table.insert(lines, string.format("Marks drawn on the pictures: %d. `mk review answers %s` reads them (in mm on lab sheets).",
			state.marks :: number, review.path))
	else
		table.insert(lines, string.format("`mk review answers %s` reads them back.", review.path))
	end
	return table.concat(lines, "\n")
end

-- Read and check the review (mk review show), then send its pictures.
local function load(state: State)
	local cx = state.cx
	local path = state.path
	if path == nil then
		return
	end
	state.loading, state.err = true, nil
	local ok, problem = pcall(function()
		tern.process.run(L.argv({ "review", "show", path }), { timeout_ms = SHOW_TIMEOUT_MS }, function(r)
			if live[cx.pane] ~= state then
				return
			end
			state.loading = false
			local rep, err = L.report(r)
			if rep == nil then
				state.err = err
				cx:render()
				return
			end
			local review = rep :: Review
			state.review = review
			state.pics, state.models = {}, {}
			local np, nm = 0, 0
			local function index(imgs: { Img }, models: { Model })
				for _, img in imgs do
					np += 1
					img.id = "i" .. np
					state.pics[img.id :: string] = img
					if state.blobs[img.path] == nil then
						local okr, data = pcall(tern.fs.read, img.show)
						if okr then
							state.blobs[img.path] = cx:blob(data :: string, L.mime(img.show))
						end
					end
				end
				for _, m in models do
					nm += 1
					m.id = "m" .. nm
					state.models[m.id :: string] = m
				end
			end
			for _, q in review.questions do
				index(q.images, q.models)
				for _, o in q.options do
					index(o.images, o.models)
				end
			end
			state.answers = {}
			if type(review.answers) == "table" then
				for qid, a in review.answers do
					if type(a) == "table" then
						state.answers[qid] = { choice = a.choice, notes = if type(a.notes) == "string" then a.notes else "" }
					end
				end
			end
			state.sent = review.sent
			state.current = math.clamp(state.current, 1, math.max(1, #review.questions))
			cx:render()
		end)
	end)
	if not ok then
		state.loading = false
		state.err = "cannot run mk: " .. tostring(problem)
	end
end

-- Show a model in the 3D block: a file it opens as it is, else posed by mk model glb first (once per block run).
local function open_model(state: State, id: string)
	local cx = state.cx
	local m = state.models[id]
	if m == nil then
		return
	end
	if m.open ~= nil then
		cx:open(L.link("preview", { path = m.open :: string }))
		return
	end
	local b = state.built[id]
	if b ~= nil and b.state == "building" then
		return
	end
	if b ~= nil and b.state == "ready" and b.glb ~= nil and tern.fs.exists(b.glb :: string) then
		cx:open(L.link("preview", { path = b.glb :: string }))
		return
	end
	local glb = m.glb
	if glb == nil then
		return
	end
	local dir = tern.plugin.data .. "/models"
	pcall(tern.fs.mkdir, dir)
	local out = dir .. "/" .. L.hash(glb) .. ".glb"
	local argv: { string } = {}                      -- decoded JSON arrays refuse table.clone (protected metatable)
	for _, a in glb do
		table.insert(argv, a)
	end
	table.insert(argv, "--out")
	table.insert(argv, out)
	state.built[id] = { state = "building" }
	local ok, problem = pcall(function()
		tern.process.run(L.argv(argv), { timeout_ms = GLB_TIMEOUT_MS }, function(r)
			if live[cx.pane] ~= state then
				return
			end
			local rep, err = L.report(r)
			if rep == nil then
				state.built[id] = { state = "error", err = err }
			else
				state.built[id] = { state = "ready", glb = out }
				cx:open(L.link("preview", { path = out }))
			end
			cx:render()
		end)
	end)
	if not ok then
		state.built[id] = { state = "error", err = "cannot run mk: " .. tostring(problem) }
	end
end

local function choose(state: State, qid: string, oid: string)
	local review = state.review
	if review == nil then
		return
	end
	for i, q in review.questions do
		if q.id == qid then
			state.current = i
			if label_of(q, oid) ~= nil then
				answer(state, qid).choice = oid
				write_answers(state)
			end
		end
	end
end

local function notes_id(qid: string): string
	return "main.q-" .. qid .. ".notes"
end

local function focus_notes(state: State, qid: string)
	state.focus = qid
	state.cursor = utf8.len(answer(state, qid).notes) or 0
	state.cx:frame({ { "focus", notes_id(qid) } })
end

local function leave_notes(state: State)
	state.focus = nil
	state.cx:frame({ { "focus", tern.json.null } })
end

-- ------------------------------------------------------------------------------------------------------------- view
local function picture(state: State, img: Img, lines: string): Node
	local blob = state.blobs[img.path]
	local pic: Node
	if blob ~= nil then
		pic = ui.node("image", { blob = blob, alt = img.name, path = img.path, max = { h = lines } })
	else
		pic = ui.text({ ui.span(img.name .. ": cannot read it", "warning") })
	end
	local tools: { Node } = { ui.text({ ui.span(img.name, "muted") }) }
	if img.sheet then
		table.insert(tools, L.props(ui.badge("lab sheet: marks in mm", "info"), { title = "Lines drawn over it come back in millimetres on the model" }))
	end
	table.insert(tools, L.button("Mark up", "markup=" .. (img.id :: string), "neutral", "markup"))
	return L.props(ui.col({ pic, L.props(ui.row(tools), { role = "mk.tools" }) }),
		{ key = img.id, role = "mk.picture", grow = 1, basis = 0 })
end

local function models_row(state: State, models: { Model }): Node?
	if #models == 0 then
		return nil
	end
	local kids: { Node } = {}
	for _, m in models do
		local id = m.id :: string
		local b = state.built[id]
		if b ~= nil and b.state == "building" then
			table.insert(kids, L.props(ui.node("spinner", { style = "dots", label = { ui.span("Posing " .. m.label .. "…", "muted") } }), { key = id }))
		else
			table.insert(kids, L.button(m.label .. " in 3D", "model=" .. id, "info", id))
			if b ~= nil and b.state == "error" then
				table.insert(kids, L.props(ui.text({ ui.span(b.err or "failed", "error") }), { key = id .. "-err" }))
			end
		end
	end
	return L.props(ui.row(kids), { role = "mk.models" })
end

local function option_view(state: State, q: Question, o: Option): Node
	local a = state.answers[q.id]
	local chosen = a ~= nil and a.choice == o.id
	local pick = ui.el("label", { class = "mk-pick" }, {
		ui.el("input", { type = "radio", name = "q-" .. q.id, value = o.id, checked = chosen }),
		ui.el("span", { class = "mk-pick-id", text = o.id }),
		ui.el("span", { class = "mk-pick-label", text = o.label }),
	})
	local kids: { Node } = { L.props(pick, { key = "pick" }) }
	if q.recommended == o.id then
		table.insert(kids, L.props(ui.badge("recommended", "accent"), { key = "rec" }))
	end
	if o.text ~= "" then
		table.insert(kids, L.props(ui.md(o.text), { key = "text" }))
	end
	for _, img in o.images do
		table.insert(kids, picture(state, img, "16lines"))
	end
	local mr = models_row(state, o.models)
	if mr ~= nil then
		table.insert(kids, L.props(mr, { key = "models" }))
	end
	return L.props(ui.col(kids), {
		key = "o-" .. o.id,
		role = "mk.option",
		tone = if chosen then "accent" else nil,
		mark = if chosen then "pick" else nil,
		grow = 1,
		basis = 0,
	})
end

local function question_view(state: State, i: number, q: Question): Node
	local a = state.answers[q.id]
	local chosen = a ~= nil and a.choice ~= nil
	local kids: { Node } = {}
	if q.text ~= "" then
		table.insert(kids, L.props(ui.md(q.text), { key = "text" }))
	end
	if #q.images > 0 then
		local pics: { Node } = {}
		for _, img in q.images do
			table.insert(pics, picture(state, img, "30lines"))
		end
		table.insert(kids, L.props(ui.row(pics), { key = "pics", role = "mk.pictures" }))
	end
	local mr = models_row(state, q.models)
	if mr ~= nil then
		table.insert(kids, L.props(mr, { key = "models" }))
	end
	if #q.options > 0 then
		local opts: { Node } = {}
		for _, o in q.options do
			table.insert(opts, option_view(state, q, o))
		end
		table.insert(kids, L.props(ui.row(opts), { key = "options", role = "mk.options" }))
	end
	local notes = if a ~= nil then a.notes else ""
	table.insert(kids, ui.node("editor", {
		key = "notes",
		role = "mk.notes",
		text = notes,
		cursor = if state.focus == q.id then L.utf16(notes, state.cursor) else nil,
		placeholder = "Notes for the agent: click here and type (Esc leaves)",
		maxLines = 8,
		aria = "Notes: " .. q.ask,
	}))
	local head: { Span | string } = { ui.span(tostring(i) .. "  ", "muted"), ui.span(q.ask, "strong") }
	if chosen and a ~= nil then
		table.insert(head, ui.span("   " .. (a.choice :: string) .. ": " .. (label_of(q, a.choice) or ""), "success"))
	end
	return L.props(ui.card(head, kids), {
		key = "q-" .. q.id,
		tone = if chosen then "success" else nil,
		mark = if i == state.current then "pick" else nil,
	})
end

local function dock_view(state: State): Node
	local n, total = answered(state)
	local line: { Node } = {
		ui.text({ ui.span(string.format("%d of %d answered", n, total), if n == total and total > 0 then "success" else "muted") }),
	}
	if #state.boards > 0 then
		local label = if state.marks ~= nil then string.format("Read marks (%d)", state.marks :: number) else "Read marks"
		table.insert(line, L.button(label, "read_marks"))
	end
	table.insert(line, L.button(if state.sent ~= nil then "Send again" else "Send to the agent", "send", "accent"))
	if state.status ~= nil then
		table.insert(line, L.props(ui.text({ ui.span(state.status :: string, "muted") }), { key = "status" }))
	end
	return ui.col({
		L.props(ui.row(line), { key = "bar" }),
		L.props(ui.text({ ui.span("j/k question · 1-9 choose · n notes (Esc leaves) · click a picture to zoom", "dim") }), { key = "keys" }),
	})
end

local block: BlockDef<State> = {
	init = function(cx, args, saved)
		local s: any = if type(saved) == "table" then saved else {}
		local state: State = {
			cx = cx,
			path = if type(s.path) == "string" then s.path else args[1],
			review = nil,
			err = nil,
			loading = false,
			blobs = {},
			pics = {},
			models = {},
			built = {},
			answers = {},
			current = if type(s.current) == "number" then s.current else 1,
			focus = nil,
			cursor = 0,
			boards = if type(s.boards) == "table" then s.boards else {},
			marks = if type(s.marks) == "number" then s.marks else nil,
			sent = nil,
			status = nil,
		}
		live[cx.pane] = state
		if state.path ~= nil and not tern.fs.exists(state.path :: string) then
			state.err = "no file " .. (state.path :: string)
		else
			load(state)
		end
		return state
	end,

	title = function(state)
		local review = state.review
		if review == nil then
			return "Review"
		end
		local n, total = answered(state)
		return string.format("%s (%d/%d)", review.title, n, total)
	end,

	view = function(state, _cx)
		if state.path == nil then
			return { main = ui.col({ ui.text({ ui.span("Open a NAME.review.toml to answer it here (mk review open)", "muted") }) }) }
		end
		if state.err ~= nil then
			return {
				main = ui.col({
					L.props(ui.card({ ui.span("This review cannot be shown", "strong") }, {
						ui.code(state.err :: string, "text"),
						ui.text({ ui.path(state.path :: string) }),
						L.button("Read it again", "reload", "accent"),
					}), { tone = "error" }),
				}),
			}
		end
		local review = state.review
		if review == nil then
			return { main = ui.col({ ui.node("spinner", { style = "dots", label = { ui.span("Reading the review…", "muted") } }) }) }
		end
		local kids: { Node } = {}
		local head: { Node } = { ui.text({ ui.span(review.title, "strong") }) }
		if review.text ~= "" then
			table.insert(head, ui.md(review.text))
		end
		table.insert(kids, L.props(ui.col(head), { key = "head", role = "mk.head" }))
		for i, q in review.questions do
			table.insert(kids, question_view(state, i, q))
		end
		return { main = L.props(ui.col(kids), { role = "mk.review" }), dock = dock_view(state) }
	end,

	key = function(state, key, _cx)
		local review = state.review
		if review == nil then
			return false
		end
		local name = key.name
		local qid = state.focus
		if qid ~= nil then
			local a = answer(state, qid)
			local len = utf8.len(a.notes) or 0
			if name == "escape" then
				leave_notes(state)
				return true
			elseif name == "enter" or name == "paste" or (key.text ~= nil and key.text ~= "" and not key.ctrl and not key.meta) then
				local text = if name == "enter" then "\n" else key.text :: string
				local b = L.byte_at(a.notes, state.cursor)
				a.notes = string.sub(a.notes, 1, b - 1) .. text .. string.sub(a.notes, b)
				state.cursor += utf8.len(text) or #text
			elseif name == "backspace" then
				if state.cursor > 0 then
					local b0, b1 = L.byte_at(a.notes, state.cursor - 1), L.byte_at(a.notes, state.cursor)
					a.notes = string.sub(a.notes, 1, b0 - 1) .. string.sub(a.notes, b1)
					state.cursor -= 1
				end
			elseif name == "delete" then
				if state.cursor < len then
					local b0, b1 = L.byte_at(a.notes, state.cursor), L.byte_at(a.notes, state.cursor + 1)
					a.notes = string.sub(a.notes, 1, b0 - 1) .. string.sub(a.notes, b1)
				end
			elseif name == "left" then
				state.cursor = math.max(0, state.cursor - 1)
				return true
			elseif name == "right" then
				state.cursor = math.min(len, state.cursor + 1)
				return true
			elseif name == "home" then
				state.cursor = 0
				return true
			elseif name == "end" then
				state.cursor = len
				return true
			else
				return false
			end
			write_answers(state)
			return true
		end
		local q = review.questions[state.current]
		if name == "j" or name == "down" then
			state.current = math.min(state.current + 1, #review.questions)
		elseif name == "k" or name == "up" then
			state.current = math.max(state.current - 1, 1)
		elseif name == "n" and q ~= nil then
			focus_notes(state, q.id)
		elseif q ~= nil and string.match(name, "^[1-9]$") ~= nil then
			local o = q.options[tonumber(name) :: number]
			if o == nil then
				return false
			end
			choose(state, q.id, o.id)
		else
			return false
		end
		return true
	end,

	event = function(state, ev, cx)
		local review = state.review
		if ev.ev == "change" then
			local name = ev.name
			if type(name) == "string" and string.sub(name, 1, 2) == "q-" and ev.checked and type(ev.value) == "string" then
				choose(state, string.sub(name, 3), ev.value)
			end
			return
		end
		if ev.ev == "focus" then
			local qid = if type(ev.id) == "string" then string.match(ev.id, "^main%.q%-(.+)%.notes$") else nil
			if qid ~= nil then
				focus_notes(state, qid)
			end
			return
		end
		if ev.ev ~= "action" then
			return
		end
		local act, value = ev.act, if type(ev.value) == "string" then ev.value else ""
		if act == "reload" then
			load(state)
		elseif review == nil then
			return
		elseif act == "model" then
			open_model(state, value)
		elseif act == "markup" then
			local img = state.pics[value]
			if img ~= nil then
				state.status = "Opening a whiteboard with " .. img.name .. "…"
				cx:open(L.link("markup", {
					image = img.path,
					show = img.show,
					w = tostring(img.w),
					h = tostring(img.h),
					pane = tostring(cx.pane),
				}))
			end
		elseif act == "board" then
			local ok, b = pcall(tern.json.decode, value)
			if ok and type(b) == "table" then
				local kept = {}
				for _, old in state.boards do
					if old.pane ~= b.pane then                 -- a pane id is reused after a restart
						table.insert(kept, old)
					end
				end
				table.insert(kept, b)
				state.boards = kept
				state.status = "Draw on the picture in the whiteboard, then Read marks"
			end
		elseif act == "read_marks" then
			state.status = "Reading the marks…"
			cx:open(L.link("marks", { boards = tern.json.encode(tern.json.array(state.boards)), pane = tostring(cx.pane) }))
		elseif act == "marks_read" then
			local ok, got = pcall(tern.json.decode, value)
			if ok and type(got) == "table" then
				local wrote, err = pcall(L.write_json, review.marks_file, { read = L.now(), images = got.images or tern.json.array({}) })
				state.marks = tonumber(got.count) or 0
				state.status = if wrote then string.format("%d marks read", state.marks :: number) else "cannot write the marks: " .. tostring(err)
			end
		elseif act == "send" then
			if state.focus ~= nil then
				leave_notes(state)
			end
			local text = message(state, review)
			state.sent = L.now()
			write_answers(state, { sent = state.sent, message = text })
			state.status = "Sending…"
			cx:open(L.link("send", {
				message = text,
				reply_to = if review.reply_to ~= nil then tostring(review.reply_to) else "",
				pane = tostring(cx.pane),
			}))
		elseif act == "sent" then
			state.status = if value == "ok" then "Sent to the agent at " .. os.date("%H:%M") else "Not sent: " .. value
		end
	end,

	save = function(state)
		return { path = state.path, current = state.current, boards = tern.json.array(state.boards), marks = state.marks }
	end,
}

return block
