import clsx from 'clsx'

import { Icon } from '@/components/Icon'

const styles = {
  warning: {
    container:
      'bg-orange-50 dark:bg-zinc-800/60 dark:ring-1 dark:ring-zinc-300/10',
    title: 'text-orange-900 dark:text-orange-400',
    body: 'text-orange-800 [--tw-prose-background:theme(colors.orange.50)] prose-a:text-orange-900 prose-code:text-orange-900 dark:text-zinc-300 dark:prose-code:text-zinc-300',
  },
  note: {
    container:
      'bg-amber-50 dark:bg-zinc-800/60 dark:ring-1 dark:ring-zinc-300/10',
    title: 'text-amber-900 dark:text-amber-500',
    body: 'text-amber-800 [--tw-prose-underline:theme(colors.amber.400)] [--tw-prose-background:theme(colors.amber.50)] prose-a:text-amber-900 prose-code:text-amber-900 dark:text-zinc-300 dark:[--tw-prose-underline:theme(colors.amber.700)] dark:prose-code:text-zinc-300',
  },
}

const icons = {
  note: (props) => <Icon icon="lightbulb" {...props} />,
  warning: (props) => <Icon icon="warning" color="amber" {...props} />,
}

export function Callout({ title, children, type = 'note' }) {
  let IconComponent = icons[type]

  return (
    <div className={clsx('my-8 flex rounded-3xl p-6', styles[type].container)}>
      <IconComponent className="h-8 w-8 flex-none" />
      <div className="ml-4 flex-auto">
        <p className={clsx('m-0 font-display text-xl', styles[type].title)}>
          {title}
        </p>
        <div className={clsx('prose mt-2.5', styles[type].body)}>
          {children}
        </div>
      </div>
    </div>
  )
}
