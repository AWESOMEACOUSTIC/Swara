"use client";

import { useEffect, useState } from "react";
import { Button } from "~/components/ui/button";
import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
} from "~/components/ui/alert-dialog";
import { Input } from "~/components/ui/input";

type RenameDialogProps = {
	track: {
		id: string;
		title: string | null;
	};
	onClose: () => void;
	onRename: (trackId: string, newTitle: string) => void | Promise<void>;
};

export function RenameDialog({ track, onClose, onRename }: RenameDialogProps) {
	const [title, setTitle] = useState(track.title ?? "");
	const [saving, setSaving] = useState(false);

	useEffect(() => {
		setTitle(track.title ?? "");
	}, [track.id, track.title]);

	const handleRename = async () => {
		const trimmed = title.trim();
		if (!trimmed) return;

		setSaving(true);
		try {
			await onRename(track.id, trimmed);
			onClose();
		} finally {
			setSaving(false);
		}
	};

	return (
		<AlertDialog open onOpenChange={(open) => !open && onClose()}>
			<AlertDialogContent>
				<AlertDialogHeader>
					<AlertDialogTitle>Rename track</AlertDialogTitle>
				</AlertDialogHeader>

				<div className="space-y-2">
					<Input
						value={title}
						onChange={(event) => setTitle(event.target.value)}
						placeholder="Track title"
					/>
				</div>

				<AlertDialogFooter>
					<AlertDialogCancel>Cancel</AlertDialogCancel>
					<AlertDialogAction asChild>
						<Button onClick={handleRename} disabled={saving}>
							{saving ? "Saving..." : "Save"}
						</Button>
					</AlertDialogAction>
				</AlertDialogFooter>
			</AlertDialogContent>
		</AlertDialog>
	);
}
