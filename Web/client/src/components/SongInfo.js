import React, {useState} from "react";
import {Table} from "reactstrap";

import Polarizing from "./Polarizing";


const SongInfo = (props) => {

    const [Likes, setLikes] = useState(new Set());
    const [Dislikes, setDislikes] = useState(new Set());

    const addSongList = (id, value) => {

        id = parseInt(id);

        deleteSong(id);

        if (value === 'like') {

            Likes.add(id);
            setLikes(Likes);
        }

        if (value === 'dislike') {

            Dislikes.add(id);
            setDislikes(Dislikes);
        }

        if (props.onSyncList) {

            props.onSyncList([...Likes], [...Dislikes]);
        }
    };

    const deleteSong = (id) => {

        if (Likes.has(id)) {

            Likes.delete(id);
            setLikes(Likes);
        }

        if (Dislikes.has(id)) {

            Dislikes.delete(id);
            setDislikes(Dislikes);
        }
    };

    const renderSongs = props.song_list && props.song_list.map((song, index) => {
        return (
            <tr key={index}>
                <th scope="row">
                    <span className="mb-0 text-sm">
                        {song.song_name}
                    </span>
                </th>
                <td>
                    {song.artist}
                </td>
                {
                    props.btn_type !== "delete" ?
                        (
                            <td>
                                {song.album_name}
                            </td>
                        ) : null
                }
                <td className="text-right">
                    <Polarizing id={song.song_id} addList={addSongList} btn_type={props.btn_type}/>
                </td>
            </tr>
        );
    });

    return (
        <Table className="align-items-center table-flush" responsive>
            <thead className="thead-light">
            <tr>
                <th scope="col">곡명</th>
                <th scope="col">아티스트</th>
                {
                    props.btn_type !== "delete" ? (<th scope="col">앨범</th>) : null
                }
                <th scope="col"/>
            </tr>
            </thead>
            {
                renderSongs ? (<tbody>{renderSongs}</tbody>) : null
            }
        </Table>
    );
}

export default SongInfo;
