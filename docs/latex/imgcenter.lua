-- pandoc lua filter for DDA-manual LaTeX build:
-- (1) wrap standalone images (one Image per paragraph) in a centered
--     adjustbox so they scale DOWN to the text width but never dominate
--     a page, and stay exactly where the prose put them (no float, so each
--     image keeps its bold "**Fig. N — ...**" caption paragraph).
--     Sentinels around the block let build_latex.py post-process the .tex
--     and weld image + caption into one unbreakable minipage; the max
--     height leaves room for that caption below (0.74, was 0.85).
-- (2) move the bold "**Table N — ...**" paragraph that the manual puts
--     directly ABOVE each pipe table into the table's own caption, so the
--     LaTeX writer emits a real \caption in the longtable head (caption
--     ABOVE the table, matching the manual's markdown layout; figure
--     captions stay BELOW).

local function is_table_caption(blk)
  if blk.t ~= 'Para' or #blk.content == 0 then return false end
  local c = blk.content
  if c[1].t == 'Strong' and #c[1].content > 0
     and c[1].content[1].t == 'Str' then
    return c[1].content[1].text == 'Table'    -- "**Table N — …**"
  end
  return false
end

function Para(el)
  if #el.content == 1 and el.content[1].t == 'Image' then
    local src = el.content[1].src
    return pandoc.RawBlock('latex',
      '%IMG-BEGIN\n\\begin{center}\\begin{adjustbox}{max width=0.97\\textwidth,' ..
      ' max height=0.74\\textheight}\\includegraphics{' .. src ..
      '}\\end{adjustbox}\\end{center}\n%IMG-END')
  end
end

function Pandoc(doc)
  local out = {}
  local i = 1
  while i <= #doc.blocks do
    local blk = doc.blocks[i]
    if blk.t == 'Table' and i > 1 and is_table_caption(doc.blocks[i - 1]) then
      table.remove(out)                    -- drop the standalone paragraph
      blk.caption.long = { doc.blocks[i - 1] }
    end
    out[#out + 1] = blk
    i = i + 1
  end
  return pandoc.Pandoc(out, doc.meta)
end
