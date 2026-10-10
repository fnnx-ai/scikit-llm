// Collects every src/app/**/page.md into the flat section list consumed by
// src/lib/search.js. Run before `next dev` / `next build` (see package.json) so
// the client search bundle can statically import the result.
import * as fs from 'node:fs'
import * as path from 'node:path'
import * as url from 'node:url'

import Markdoc from '@markdoc/markdoc'
import { slugifyWithCounter } from '@sindresorhus/slugify'
import glob from 'fast-glob'

const rootDir = path.join(url.fileURLToPath(import.meta.url), '../..')
const appDir = path.join(rootDir, 'src/app')
const outputFile = path.join(rootDir, 'src/lib/search-data.json')

const slugify = slugifyWithCounter()

function toString(node) {
  let str =
    node.type === 'text' && typeof node.attributes?.content === 'string'
      ? node.attributes.content
      : ''
  if ('children' in node) {
    for (let child of node.children) {
      str += toString(child)
    }
  }
  return str
}

// Sections are [title, hash, paragraphs]; the first one is the page itself and
// carries a null hash so it links to the bare page URL.
function extractSections(node, sections, isRoot = true) {
  if (isRoot) {
    slugify.reset()
  }
  if (node.type === 'heading' || node.type === 'paragraph') {
    let content = toString(node).trim()
    if (node.type === 'heading' && node.attributes.level <= 2) {
      let hash = node.attributes?.id ?? slugify(content)
      sections.push([content, hash, []])
    } else {
      sections.at(-1)[2].push(content)
    }
  } else if ('children' in node) {
    for (let child of node.children) {
      extractSections(child, sections, false)
    }
  }
}

const pages = glob
  .sync('**/page.md', { cwd: appDir })
  .sort()
  .map((file) => {
    let url = file === 'page.md' ? '/' : `/${file.replace(/\/page\.md$/, '')}`
    let ast = Markdoc.parse(fs.readFileSync(path.join(appDir, file), 'utf8'))
    let title = ast.attributes?.frontmatter?.match(/^title:\s*(.*?)\s*$/m)?.[1]
    let sections = [[title ?? url, null, []]]
    extractSections(ast, sections)
    return { url, sections }
  })

fs.mkdirSync(path.dirname(outputFile), { recursive: true })
fs.writeFileSync(outputFile, JSON.stringify(pages))

console.log(
  `Indexed ${pages.length} page${pages.length === 1 ? '' : 's'} for search.`,
)
