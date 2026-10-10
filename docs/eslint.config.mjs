import next from 'eslint-config-next/core-web-vitals'

const config = [{ ignores: ['.next/**', 'out/**'] }, ...next]

export default config
