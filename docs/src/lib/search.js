import { Document } from 'flexsearch'

import pages from './search-data.json'

const sectionIndex = new Document({
  tokenize: 'full',
  document: {
    id: 'url',
    index: 'content',
    store: ['title', 'pageTitle'],
  },
  context: {
    resolution: 9,
    depth: 2,
    bidirectional: true,
  },
})

for (let { url, sections } of pages) {
  for (let [title, hash, content] of sections) {
    sectionIndex.add({
      url: url + (hash ? '#' + hash : ''),
      title,
      content: [title, ...content].join('\n'),
      pageTitle: hash ? sections[0][0] : undefined,
    })
  }
}

export function search(query, options = {}) {
  let result = sectionIndex.search(query, { ...options, enrich: true })

  if (result.length === 0) {
    return []
  }

  return result[0].result.map((item) => ({
    url: item.id,
    title: item.doc.title,
    pageTitle: item.doc.pageTitle,
  }))
}
