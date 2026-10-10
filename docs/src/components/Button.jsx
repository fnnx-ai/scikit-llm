import Link from 'next/link'
import clsx from 'clsx'

const variantStyles = {
  primary:
    'rounded-full bg-orange-300 py-2 px-4 text-sm font-semibold text-zinc-900 hover:bg-orange-200 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-orange-300/50 active:bg-orange-500',
  secondary:
    'rounded-full bg-zinc-800 py-2 px-4 text-sm font-medium text-white hover:bg-zinc-700 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white/50 active:text-zinc-400',
}

export function Button({ variant = 'primary', className, ...props }) {
  className = clsx(variantStyles[variant], className)

  return typeof props.href === 'undefined' ? (
    <button className={className} {...props} />
  ) : (
    <Link className={className} {...props} />
  )
}
