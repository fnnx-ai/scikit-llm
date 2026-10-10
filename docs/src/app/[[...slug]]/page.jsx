import * as React from 'react'
import { notFound } from 'next/navigation'
import { renderers } from '@markdoc/markdoc'

import { getPage, getPageSlugs } from '@/lib/pages'
import { site } from '@/lib/site'

export const dynamicParams = false

export function generateStaticParams() {
  return getPageSlugs().map((slug) => ({ slug }))
}

export async function generateMetadata({ params }) {
  const { slug = [] } = await params
  const frontmatter = getPage(slug)?.frontmatter ?? {}

  return {
    title: slug.length === 0 ? { absolute: site.name } : frontmatter.title,
    description: frontmatter.description ?? site.description,
  }
}

export default async function Page({ params }) {
  const { slug = [] } = await params
  const page = getPage(slug)

  if (!page) {
    notFound()
  }

  return renderers.react(page.content, React)
}
