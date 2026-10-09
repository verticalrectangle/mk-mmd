--!strict
-- The mk plugin's window half: NAME.review.toml and .pmx files open in mk's blocks, and the mk:// links those blocks
-- follow (their `cx:open`) do what only a window can: show a .glb in the 3D block (preview), post a review's answers to
-- the agent that opened it (send), put a picture on a whiteboard to draw on (markup), and read back what was drawn
-- (marks). Window calls have a 50 ms budget, so this half reads and writes no files: a link carries what it needs, the
-- slow parts run in timer ticks of their own, and results go back to the block as actions on its dock (`sent`, `board`,
-- `marks_read`), whose 2 s host budget does the writing.

local L = require("./lib")

local PMX_FROM = { files = true, palette = true, link = true, system = true, cli = true, drop = true }

tern.route.open(function(req: OpenRequest, _cx: WindowCx): RouteDecision?
	local p = string.lower(req.path)
	if string.sub(p, -12) == ".review.toml" and req.how ~= "replace" then
		return { block = "mk.review", args = { req.path } }
	end
	if string.sub(p, -4) == ".pmx" and PMX_FROM[req.origin] then
		return { block = "mk.pmx", args = { req.path } }
	end
	return nil
end)

-- Tell a block what happened: an action on its dock row, which its `event` handler hears.
local function poke(cx: WindowCx, pane: string?, act: string, value: string)
	local n = tonumber(pane or "")
	if n == nil then
		return
	end
	pcall(function()
		cx.session:event(n :: number, { ev = "action", id = "dock.bar", act = act, value = value })
	end)
end

local function send(cx: WindowCx, q: { [string]: string })
	local target: number? = tonumber(q.reply_to or "")
	if target == nil then
		local agents = cx.agents:list()
		if #agents == 1 then
			target = agents[1].pane
		end
	end
	if target == nil then
		cx:toast("error", "mk review: which agent?", "Open the review with `mk review open` in the agent's pane")
		poke(cx, q.pane, "sent", "no agent named (mk review open names the pane it runs in)")
		return
	end
	local ok, err = pcall(function()
		cx.agents:ask(target :: number, q.message or "")
	end)
	if ok then
		cx:toast("success", "Answers sent to the agent")
		poke(cx, q.pane, "sent", "ok")
	else
		cx:toast("error", "mk review: could not send", tostring(err))
		poke(cx, q.pane, "sent", tostring(err))
	end
end

local function markup(cx: WindowCx, q: { [string]: string })
	local w, h = tonumber(q.w) or 1000, tonumber(q.h) or 1000
	local image, show, block = q.image, q.show or q.image, q.pane
	cx.whiteboard:open({ how = "beside" }):next(function(pane: number?, err: string?, wcx: WindowCx?)
		if wcx == nil then
			return
		end
		if pane == nil then
			wcx:toast("error", "mk review: no whiteboard", err)
			return
		end
		local ok, res = pcall(function()
			return wcx.whiteboard:edit(pane, {
				{ op = "add", type = "image", path = show, x = 0, y = 0, w = w, ref = "picture" },
				{ op = "lock", ids = { "picture" } },
				{ op = "show", x = 0, y = 0, w = w, h = h },
			})
		end)
		if not ok then
			wcx:toast("error", "mk review: cannot put the picture on the whiteboard", tostring(res))
			return
		end
		poke(wcx, block, "board", tern.json.encode({ pane = pane, shape = res.refs.picture, image = image, w = w, h = h }))
		wcx:toast("info", "Draw on the picture", "Lines and arrows come back measured, boxes and text with their words. Then Read marks.")
	end)
end

local function centre(s: any): { number }
	return { s.x + s.w / 2, s.y + s.h / 2 }
end

-- The marks on one board, in its picture's own pixels; nil when the board or its picture is gone.
local function board_marks(cx: WindowCx, b: any): any
	local ok, wb = pcall(function()
		return cx.whiteboard:read(b.pane)
	end)
	if not ok or wb == nil then
		return nil
	end
	local byid: { [string]: any } = {}
	for _, s in wb.shapes do
		byid[s.id] = s
	end
	local pic = byid[b.shape]
	if pic == nil or pic.w <= 0 or pic.h <= 0 then
		return nil
	end
	local sx, sy = b.w / pic.w, b.h / pic.h
	local function px(p: { number }): { number }
		return { (p[1] - pic.x) * sx, (p[2] - pic.y) * sy }
	end
	local function box(s: any): { number }
		return { (s.x - pic.x) * sx, (s.y - pic.y) * sy, s.w * sx, s.h * sy }
	end
	local marks = {}
	for _, s in wb.shapes do
		if s.id ~= b.shape then
			local mark: { [string]: any }
			local a = s.from_point or (if s.from_shape and byid[s.from_shape] then centre(byid[s.from_shape]) else nil)
			local z = s.to_point or (if s.to_shape and byid[s.to_shape] then centre(byid[s.to_shape]) else nil)
			if (s.type == "line" or s.type == "arrow") and a ~= nil and z ~= nil then
				mark = { kind = "line", from = px(a), to = px(z) }
			else
				mark = { kind = if s.type == "text" then "text" else "box", shape = s.type, box = box(s) }
			end
			mark.text = s.text or ""
			mark.color = s.color
			table.insert(marks, mark)
		end
	end
	return { path = b.image, w = b.w, h = b.h, board = b.pane, marks = tern.json.array(marks) }
end

-- Every board of a review, one per timer tick (each tick has its own budget), then the lot back to the block.
local function read_marks(q: { [string]: string })
	local ok, boards = pcall(tern.json.decode, q.boards or "[]")
	local list: { any } = if ok and type(boards) == "table" then boards else {}
	local images, count = {}, 0
	local function step(i: number)
		tern.timer(0, function(wcx: WindowCx?)
			if wcx == nil then
				return
			end
			local b = list[i]
			if b == nil then
				poke(wcx, q.pane, "marks_read", tern.json.encode({ count = count, images = tern.json.array(images) }))
				return
			end
			local img = board_marks(wcx, b)
			if img ~= nil then
				table.insert(images, img)
				count += #img.marks
			end
			step(i + 1)
		end)
	end
	step(1)
end

tern.route.link(function(link: LinkRequest, cx: WindowCx): RouteDecision?
	local action, q = L.parse(link.url)
	if action == nil then
		return nil
	end
	if action == "preview" and q.path ~= nil then
		cx:open(q.path, "preview")
	elseif action == "send" then
		tern.timer(0, function(wcx: WindowCx?)
			if wcx ~= nil then
				send(wcx, q)
			end
		end)
	elseif action == "markup" then
		markup(cx, q)
	elseif action == "marks" then
		read_marks(q)
	else
		cx:toast("error", "mk: unknown link", link.url)
	end
	return { handled = true }
end)
