--!strict
-- The mk plugin's window half: NAME.review.toml and .pmx files open in mk's blocks (a review file becomes the review page
-- mk serves, see review.luau), and the mk:// link the PMX block follows (its `cx:open`) shows a .glb in the 3D block
-- (preview), which only a window can do.

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

tern.route.link(function(link: LinkRequest, cx: WindowCx): RouteDecision?
	local action, q = L.parse(link.url)
	if action == nil then
		return nil
	end
	if action == "preview" and q.path ~= nil then
		cx:open(q.path, "preview")
	else
		cx:toast("error", "mk: unknown link", link.url)
	end
	return { handled = true }
end)
