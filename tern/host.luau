--!strict
-- The mk plugin's host half: the review launcher (review.luau) and the PMX block (pmx.luau).

tern.block.define("review", require("./review"))
tern.block.define("pmx", require("./pmx"))
