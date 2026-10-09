--!strict
-- Shared by both halves of the mk plugin: how to run mk, the mk:// links a block uses to reach the window half (every
-- `cx:open` a block makes goes through `tern.route.link`), JSON files, small text helpers.

local M = {}

-- mk from the mk-mmd checkout this plugin sits in (tern/, linked with `tern plugin link`), else `mk` on PATH (an
-- installed copy of the plugin, with mk installed as a tool).
function M.argv(args: { string }): { string }
	local repo = string.match(tern.plugin.dir, "^(.*)/[^/]+/?$")
	local out: { string }
	if repo ~= nil and tern.fs.exists(repo .. "/pyproject.toml") and tern.fs.exists(repo .. "/mkmmd/review.py") then
		out = { "uv", "run", "--quiet", "--project", repo, "mk" }
	else
		out = { "mk" }
	end
	for _, a in args do
		table.insert(out, a)
	end
	return out
end

function M.encode(s: string): string
	return (string.gsub(s, "[^%w%-%._~/]", function(c: string): string
		return string.format("%%%02X", string.byte(c))
	end))
end

function M.decode(s: string): string
	return (string.gsub(s, "%%(%x%x)", function(h: string): string
		return string.char(tonumber(h, 16) :: number)
	end))
end

-- `mk://ACTION?key=value&...`: what a block asks the window half to do.
function M.link(action: string, params: { [string]: string }): string
	local keys = {}
	for k in params do
		table.insert(keys, k)
	end
	table.sort(keys)
	local parts = {}
	for _, k in keys do
		table.insert(parts, k .. "=" .. M.encode(params[k]))
	end
	return "mk://" .. action .. "?" .. table.concat(parts, "&")
end

function M.parse(url: string): (string?, { [string]: string })
	local params: { [string]: string } = {}
	local action, query = string.match(url, "^mk://([%w_%-]+)%??(.*)$")
	if action == nil then
		return nil, params
	end
	for pair in string.gmatch(query or "", "[^&]+") do
		local k, v = string.match(pair, "^([^=]*)=?(.*)$")
		if k ~= nil and k ~= "" then
			params[M.decode(k)] = M.decode(v or "")
		end
	end
	return action, params
end

function M.read_json(path: string): any
	if not tern.fs.exists(path) then
		return nil
	end
	local ok, value = pcall(function()
		return tern.json.decode(tern.fs.read(path))
	end)
	return if ok then value else nil
end

function M.write_json(path: string, value: any)
	tern.fs.write(path, tern.json.encode(value, true))
end

-- What mk printed: its JSON report, or nil and why it failed (its `error`, else stderr).
function M.report(r: ProcessResult): (any, string?)
	if r.timed_out then
		return nil, "timed out"
	end
	local ok, value = pcall(tern.json.decode, r.stdout)
	if ok and type(value) == "table" then
		if r.status ~= 0 then
			return nil, tostring(value.error or r.stderr)
		end
		return value, nil
	end
	local err = if r.stderr ~= "" then r.stderr else r.stdout
	if err == "" then
		err = "mk exited with status " .. tostring(r.status)
	end
	return nil, (string.gsub(err, "%s+$", ""))
end

-- A short stable name for a list of strings (cache file names).
function M.hash(parts: { string }): string
	local h1, h2 = 5381, 52711
	for _, s in parts do
		for i = 1, #s do
			local c = string.byte(s, i)
			h1 = (h1 * 33 + c) % 4294967296
			h2 = (h2 * 31 + c) % 4294967296
		end
		h1 = (h1 * 33 + 1) % 4294967296
	end
	return string.format("%08x%08x", h1, h2)
end

function M.mime(path: string): string
	local ext = string.lower(string.match(path, "%.(%w+)$") or "")
	if ext == "jpg" or ext == "jpeg" then
		return "image/jpeg"
	elseif ext == "webp" then
		return "image/webp"
	elseif ext == "gif" then
		return "image/gif"
	end
	return "image/png"
end

function M.now(): string
	return os.date("!%Y-%m-%dT%H:%M:%SZ") :: string
end

-- A clickable chip: `act` (with `=value`) reaches the block's `event` as an action. `key` keeps its node id among its
-- siblings (default the action, which must then be unique among them).
function M.button(label: string, act: string, tone: string?, key: string?): Node
	local b = tern.ui.badge(label, tone or "neutral")
	local p: { [string]: any } = b.p or {}
	p.actions = { click = act }
	p.key = key or act
	b.p = p
	return b
end

function M.props(node: Node, props: { [string]: any }): Node
	local p: { [string]: any } = node.p or {}
	for k, v in props do
		p[k] = v
	end
	node.p = p
	return node
end

-- Whether `list` holds `value` (decoded JSON arrays refuse table.find and table.clone: protected metatables).
function M.has(list: { any }, value: any): boolean
	for _, v in list do
		if v == value then
			return true
		end
	end
	return false
end

-- UTF-8 text edited by character: the byte where character `chars` + 1 starts, and the UTF-16 offset Tern's fields
-- take for a caret after `chars` characters.
function M.byte_at(s: string, chars: number): number
	return utf8.offset(s, chars + 1) or (#s + 1)
end

function M.utf16(s: string, chars: number): number
	local units, n = 0, 0
	for _, c in utf8.codes(s) do
		if n >= chars then
			break
		end
		units += if c > 0xFFFF then 2 else 1
		n += 1
	end
	return units
end

return M
