'use client'

import { Fragment } from 'react'
import { Highlight } from 'prism-react-renderer'

const codeTheme = {
  plain: { color: '#e4e4e7', fontStyle: 'normal' },
  styles: [
    {
      types: ['builtin', 'changed', 'keyword', 'function'],
      style: { color: '#ff7500' },
    },
    {
      types: ['string', 'number', 'definition', 'boolean', 'class-name'],
      style: { color: '#fdba74' },
    },
    { types: ['comment'], style: { color: '#6a9955' } },
    {
      types: ['property', 'punctuation', 'operator'],
      style: { color: '#a1a1aa' },
    },
  ],
}

// `language` is undefined for fences opened without an info string (```), and
// prism-react-renderer requires a string.
export function Fence({ children, language = 'text' }) {
  return (
    <Highlight code={children.trimEnd()} language={language} theme={codeTheme}>
      {({ className, style, tokens, getTokenProps }) => (
        <pre className={className} style={style}>
          <code>
            {tokens.map((line, lineIndex) => (
              <Fragment key={lineIndex}>
                {line
                  .filter((token) => !token.empty)
                  .map((token, tokenIndex) => (
                    <span key={tokenIndex} {...getTokenProps({ token })} />
                  ))}
                {'\n'}
              </Fragment>
            ))}
          </code>
        </pre>
      )}
    </Highlight>
  )
}
