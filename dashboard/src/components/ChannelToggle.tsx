// dashboard/src/components/ChannelToggle.tsx
import { useState, useEffect, useCallback } from 'react'

interface Channel {
  name: string
  connected: boolean
  broadcast: boolean
  last_chat_id: number | string | null
}

const CHANNEL_ICONS: Record<string, string> = {
  telegram: '✈',
  discord: '💬',
  kakao: '💛',
  slack: '⚡',
  naver: '🟢',
}

export function ChannelToggle() {
  const [channels, setChannels] = useState<Channel[]>([])
  const [open, setOpen] = useState(false)

  const fetchChannels = useCallback(() => {
    fetch('/api/channels')
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setChannels(data) })
      .catch(() => {})
  }, [])

  useEffect(() => {
    fetchChannels()
    const id = setInterval(fetchChannels, 5000)
    return () => clearInterval(id)
  }, [fetchChannels])

  const toggleBroadcast = useCallback(async (name: string, current: boolean) => {
    await fetch(`/api/channels/${name}/broadcast`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !current }),
    })
    fetchChannels()
  }, [fetchChannels])

  if (channels.length === 0) {
    return (
      <div className="channel-toggle-wrap">
        <div className="channel-badge empty" title="No channels connected">
          <span className="channel-icon">📡</span>
          <span className="channel-label">No channels</span>
        </div>
      </div>
    )
  }

  return (
    <div className="channel-toggle-wrap">
      <button
        className="channel-toggle-btn"
        onClick={() => setOpen(!open)}
        title="Channel broadcast settings"
      >
        <span className="channel-icon">📡</span>
        <span className="channel-count">{channels.length}</span>
        <span className="channel-arrow">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="channel-dropdown">
          {channels.map(ch => (
            <div key={ch.name} className="channel-item">
              <span className="channel-item-icon">
                {CHANNEL_ICONS[ch.name] || '📨'}
              </span>
              <span className="channel-item-name">{ch.name}</span>
              {ch.last_chat_id && (
                <span className="channel-item-active" title="Has active chat">●</span>
              )}
              <button
                className={`channel-switch ${ch.broadcast ? 'on' : 'off'}`}
                onClick={() => toggleBroadcast(ch.name, ch.broadcast)}
                title={ch.broadcast ? 'Broadcasting ON' : 'Broadcasting OFF'}
              >
                <span className="channel-switch-knob" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
