import * as fs from 'node:fs'
import * as path from 'node:path'

import Markdoc from '@markdoc/markdoc'
import glob from 'fast-glob'
import { load as loadYaml } from 'js-yaml'

import nodes from '@/markdoc/nodes'
import tags from '@/markdoc/tags'

const appDir = path.join(process.cwd(), 'src/app')

export function getPageSlugs() {
  return glob
    .sync('**/page.md', { cwd: appDir })
    .sort()
    .map((file) =>
      file === 'page.md' ? [] : file.slice(0, -'/page.md'.length).split('/'),
    )
}

export function getPage(slug) {
  if (slug.some((segment) => segment.includes('.') || segment.includes('/'))) {
    return null
  }

  const file = path.join(appDir, ...slug, 'page.md')
  if (!fs.existsSync(file)) {
    return null
  }

  const ast = Markdoc.parse(fs.readFileSync(file, 'utf8'))
  const frontmatter = ast.attributes.frontmatter
    ? loadYaml(ast.attributes.frontmatter)
    : {}

  return {
    frontmatter,
    content: Markdoc.transform(ast, {
      nodes,
      tags,
      variables: { markdoc: { frontmatter } },
    }),
  }
}
