import clsx from 'clsx'

export function Prose({ as, className, ...props }) {
  let Component = as ?? 'div'

  return (
    <Component
      className={clsx(
        className,
        'prose prose-zinc max-w-none dark:prose-invert dark:text-zinc-400',
        // headings
        'prose-headings:scroll-mt-28 prose-headings:font-display prose-headings:font-normal lg:prose-headings:scroll-mt-[8.5rem]',
        'prose-lead:text-zinc-500 dark:prose-lead:text-zinc-400',
        // links
        'prose-a:font-semibold dark:prose-a:text-orange-400',
        // link underline
        'prose-a:no-underline prose-a:shadow-[inset_0_-2px_0_0_var(--tw-prose-background,#fff),inset_0_calc(-1*(var(--tw-prose-underline-size,4px)+2px))_0_0_var(--tw-prose-underline,theme(colors.orange.300))] hover:prose-a:[--tw-prose-underline-size:6px] dark:[--tw-prose-background:theme(colors.zinc.900)] dark:prose-a:shadow-[inset_0_calc(-1*var(--tw-prose-underline-size,2px))_0_0_var(--tw-prose-underline,theme(colors.orange.800))] dark:hover:prose-a:[--tw-prose-underline-size:6px]',
        // pre
        'prose-pre:rounded-xl prose-pre:bg-zinc-900 prose-pre:shadow-lg dark:prose-pre:bg-zinc-700/40 dark:prose-pre:shadow-none dark:prose-pre:ring-1 dark:prose-pre:ring-zinc-100/10',
        // hr
        'dark:prose-hr:border-zinc-800',
      )}
      {...props}
    />
  )
}
