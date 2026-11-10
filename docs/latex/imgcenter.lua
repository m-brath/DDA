-- pandoc lua filter for DDA-manual LaTeX build:
-- wrap standalone images (one Image per paragraph) in a centered
-- adjustbox so they scale DOWN to the text width but never dominate
-- a full page, and stay exactly where the prose put them (no float,
-- so each image keeps its bold "**Fig. N — ...**" caption paragraph).
function Para(el)
  if #el.content == 1 and el.content[1].t == 'Image' then
    local src = el.content[1].src
    return pandoc.RawBlock('latex',
      '\\begin{center}\\begin{adjustbox}{max width=0.97\\textwidth,' ..
      ' max height=0.85\\textheight}\\includegraphics{' .. src ..
      '}\\end{adjustbox}\\end{center}')
  end
end
