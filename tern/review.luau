--!strict
-- Block mk.review: a NAME.review.toml opened in Tern (its Files pane, `tern open`) becomes the review page mk serves
-- (mkmmd/site, docs/design.md: The site): this block runs `mk review open --replace` so the page takes its place, and
-- says why when it cannot. Agents open reviews with `mk review open` themselves, beside their own pane.

local ui = tern.ui
local L = require("./lib")

local OPEN_TIMEOUT_MS = 120000

type State = { cx: BlockCx, path: string?, err: string? }

local block: BlockDef<State> = {
	init = function(cx, args, saved)
		local s: any = if type(saved) == "table" then saved else {}
		local state: State = { cx = cx, path = if type(s.path) == "string" then s.path else args[1], err = nil }
		local path = state.path
		if path ~= nil then
			local ok, problem = pcall(function()
				tern.process.run(L.argv({ "review", "open", path, "--replace", tostring(cx.pane) }), { timeout_ms = OPEN_TIMEOUT_MS }, function(r)
					local _, err = L.report(r)
					if err ~= nil then
						state.err = err
						cx:render()
					end
				end)
			end)
			if not ok then
				state.err = "cannot run mk: " .. tostring(problem)
			end
		end
		return state
	end,

	title = function(_state)
		return "Review"
	end,

	view = function(state, _cx)
		if state.path == nil then
			return { main = ui.col({ ui.text({ ui.span("Open a NAME.review.toml to answer it (mk review open)", "muted") }) }) }
		end
		if state.err ~= nil then
			return {
				main = ui.col({
					L.props(ui.card({ ui.span("This review cannot be opened", "strong") }, {
						ui.code(state.err :: string, "text"),
						ui.text({ ui.path(state.path :: string) }),
					}), { tone = "error" }),
				}),
			}
		end
		return { main = ui.col({ ui.node("spinner", { style = "dots", label = { ui.span("Opening the review page…", "muted") } }) }) }
	end,

	save = function(state)
		return { path = state.path }
	end,
}

return block
